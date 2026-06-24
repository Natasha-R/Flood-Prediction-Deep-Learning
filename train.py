import modelling.data_pipeline as data_pipeline
import torch
from tqdm import tqdm
import time
import os 
import modelling.utils as utils
import argparse
import pandas as pd
from visualise import plot_losses
from metrics import calculate_metrics
import torch.distributed as dist
import torch.multiprocessing as mp
from segmentation_models_pytorch.losses import DiceLoss

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config_path, data_folder, modelling_folder, pretrained_path, compute_losses=False, ddp=False):

    if ddp:
        setup(rank, world_size)
        torch.cuda.manual_seed_all(47)
    else:
        torch.manual_seed(47)

    try:
        logger = utils.get_logger()
        config, config_name = utils.load_config(config_path, logger)
        model = utils.load_model(config, rank, ddp, logger, pretrained_path)
        subsets = list(pd.read_csv(f"{data_folder}/subsets/{config['data_subset_file']}.csv")["subset"].unique())
        subsets = [subset for subset in subsets if "test" not in subset]
        data_loaders = {subset: data_pipeline.create_data_loader(config, data_folder, ddp, subset, training=True) for subset in subsets}
        if config.get("loss_function", "cross entropy").lower()=="dice":
            loss_function = DiceLoss(mode="binary", from_logits=True, ignore_index=0)
        else:
            loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(config["class_weights"], dtype=torch.float32).to(rank), reduction="mean", ignore_index=0, size_average=True)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        if config.get("use_scheduler", True):
            use_scheduler = True
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=config["learning_rate"], epochs=config["number_epochs"], steps_per_epoch=len(data_loaders["train"]))
        else:
            use_scheduler = False
            
        losses = {f"total_{subset}_losses": [] for subset in subsets}
        losses.update({f"epoch_{subset}_losses": [] for subset in subsets})
        losses.update({f"epoch_{subset}_local_losses": [] for subset in subsets})
        losses.update({f"total_{subset}_local_losses": [] for subset in subsets})

        for epoch in range(1, config["number_epochs"]+1):

            epoch_start_time = time.time()
            losses.update({f"epoch_{subset}_losses": [] for subset in subsets})
            losses.update({f"epoch_{subset}_local_losses": [] for subset in subsets})
            if ddp:
                data_loaders["train"].sampler.set_epoch(epoch)

            model.train()
            for data in tqdm(data_loaders["train"], desc="Training", leave=False):
                optimizer.zero_grad()
                for item in data.keys():
                    data[item] = data[item].to(rank) #BCHW (features) #BHW (label)
                model_output = model(data)#BclassesHW

                loss = 0.0
                local_losses = 0.0
                for scale in config["output_scales"]:
                    indiv_loss = loss_function(model_output[f"{scale}_pred"].squeeze(), data[f"{scale}_label"])
                    loss = loss + config[f"{scale}_weight"] * indiv_loss
                    if scale == "local":
                        local_losses += indiv_loss.item()
                loss.backward()
                optimizer.step()
                losses["epoch_train_losses"].append(loss.item())
                losses["epoch_train_local_losses"].append(local_losses)
                if use_scheduler:
                    scheduler.step()
            
            if compute_losses:
                model.eval()
                with torch.no_grad():
                    for validation_loader in [val_set for val_set in subsets if "val" in val_set]:
                        for data in tqdm(data_loaders[validation_loader], desc=f"Validation ({validation_loader}) loss", leave=False):
                            for item in data.keys():
                                data[item] = data[item].to(rank) #BCHW (features) #BHW (label)
                            model_output = model(data)
                            loss = 0.0
                            local_losses = 0.0
                            for scale in config["output_scales"]:
                                indiv_loss = loss_function(model_output[f"{scale}_pred"], data[f"{scale}_label"])
                                loss = loss + config[f"{scale}_weight"] * indiv_loss
                                if scale == "local":
                                    local_losses += indiv_loss.item()
                            losses[f"epoch_{validation_loader}_losses"].append(loss.item())
                            losses[f"epoch_{validation_loader}_local_losses"].append(local_losses)

            loss_string = f"GPU: {rank} | Epoch {epoch} ({time.time() - epoch_start_time:.0f} seconds)"
            loss_subsets = subsets if compute_losses else ["train"]
            for loss_type in loss_subsets:
                losses[f"total_{loss_type}_losses"].append(sum(losses[f"epoch_{loss_type}_losses"]) / len(losses[f"epoch_{loss_type}_losses"]))
                loss_string += f" | {loss_type} loss: {losses[f'total_{loss_type}_losses'][-1]:.3f}"
                losses[f"total_{loss_type}_local_losses"].append(sum(losses[f"epoch_{loss_type}_local_losses"]) / len(losses[f"epoch_{loss_type}_local_losses"]))
                loss_string += f" | {loss_type} local loss: {losses[f'total_{loss_type}_local_losses'][-1]:.3f}"
            logger.info(loss_string)

            if (config["save_model_on_epoch"] != 0) and ((epoch % config["save_model_on_epoch"]) == 0) and ((rank==0) or (ddp==False)):

                model_save_path = f"{modelling_folder}/models/{config_name}_{epoch}.pth"
                torch.save(model.module.state_dict() if ddp else model.state_dict(), model_save_path)
                logger.info(f"Saved the model to: {model_save_path}")

                if compute_losses:
                    plot_losses(losses, config, config_name, modelling_folder, rank, logger)

                for validation_loader in [val_set for val_set in subsets if "val" in val_set]:
                    calculate_metrics(config, config_name, model, data_loaders[validation_loader], modelling_folder, epoch=epoch, device=rank, subset=validation_loader)
                    logger.info(f"Saved {validation_loader} evaluation metrics to: {modelling_folder}/metrics/{config_name}.csv")
    finally:
        if ddp:
            cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('-c', '--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('-d', '--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('-m', '--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('-g', '--gpu', default="ddp", help="Specify which gpu to train on. 'ddp' for parallel training, '0', '1' for a GPU, and 'cpu' for CPU.")
    parser.add_argument('-p', '--pretrained', default=False, help="A path to pretrained model weights.")
    parser.add_argument('-l', '--losses', action="store_true", default=False, help="Computes the losses after every epoch, if turned on.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    args.gpu = int(args.gpu) if args.gpu != "cpu" and args.gpu != "ddp" else args.gpu
    
    world_size = torch.cuda.device_count()
    if args.gpu=="ddp":
        mp.spawn(train, args=(world_size, args.config_path, args.data_folder, args.modelling_folder, args.pretrained, args.losses, True), nprocs=world_size, join=True)
    else:
        train(rank=args.gpu, world_size=world_size, config_path=args.config_path, data_folder=args.data_folder, 
              modelling_folder=args.modelling_folder, pretrained_path=args.pretrained, compute_losses=args.losses, ddp=False)