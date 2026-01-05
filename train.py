import modelling.data_pipeline as data_pipeline
import torch
from tqdm import tqdm
import time
import os 
import modelling.utils as utils
import argparse
import pandas as pd
from visualise import plot_losses
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config_path, data_folder, modelling_folder, ddp=False):

    if ddp:
        setup(rank, world_size)

    logger = utils.get_logger()
    config, config_name = utils.load_config(config_path, logger)
    model = utils.load_model(config, rank, logger, ddp)
    subsets = list(pd.read_csv(f"{data_folder}/metadata/{config['data_subset_file']}.csv")["subset"].unique())
    data_loaders = {subset: data_pipeline.create_data_loader(config, data_folder, ddp, subset) for subset in subsets}
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0, size_average=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=config["learning_rate"], total_steps=config["number_epochs"])

    losses = {f"total_{subset}_losses": [] for subset in subsets}
    losses.update({f"epoch_{subset}_losses": [] for subset in subsets})

    for epoch in range(1, config["number_epochs"]+1):

        epoch_start_time = time.time()
        losses.update({f"epoch_{subset}_losses": [] for subset in subsets})
        if ddp:
            data_loaders["train"].sampler.set_epoch(epoch)

        model.train()
        for data in tqdm(data_loaders["train"], desc="Training", leave=False):
            optimizer.zero_grad()
            for item in data.keys():
                data[item] = data[item].to(rank) #BCHW (features) #BHW (label)
            model_output = model(data) #BclassesHW
            loss = 0.0
            for scale in config["scales"]:
                loss = loss + config[f"{scale}_weight"] * loss_function(model_output[f"{scale}_pred"], data[f"{scale}_label"])
            loss.backward()
            optimizer.step()
            losses["epoch_train_losses"].append(loss.item())
            
        model.eval()
        with torch.no_grad():
            for validation_loader in [val_set for val_set in subsets if "val" in val_set]:
                for data in tqdm(data_loaders[validation_loader], desc=f"Validation ({validation_loader}) loss", leave=False):
                    for item in data.keys():
                        data[item] = data[item].to(rank) #BCHW (features) #BHW (label)
                    model_output = model(data)
                    loss = 0.0
                    for scale in config["scales"]:
                        loss = loss + config[f"{scale}_weight"] * loss_function(model_output[f"{scale}_pred"], data[f"{scale}_label"])
                    losses[f"epoch_{validation_loader}_losses"].append(loss.item())

        scheduler.step()

        loss_string = f"Rank: {rank} | Epoch {epoch} ({time.time() - epoch_start_time:.0f} seconds)"
        for loss_type in subsets:
            losses[f"total_{loss_type}_losses"].append(sum(losses[f"epoch_{loss_type}_losses"]) / len(losses[f"epoch_{loss_type}_losses"]))
            loss_string += f" | {loss_type} loss: {losses[f'total_{loss_type}_losses'][-1]:.3f}"
        logger.info(loss_string)

        if config["save_model_on_epoch"] != 0 and ((epoch % config["save_model_on_epoch"]) == 0) and (rank==0):
            model_save_path = f"{modelling_folder}/models/{config_name}_{epoch}.pth"
            torch.save(model.module.state_dict() if ddp else model.state_dict(), model_save_path)
            logger.info(f"Saved the model to: {model_save_path}")

    plot_losses(losses, config_name, modelling_folder, rank, logger)

    if ddp:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('--gpu', default="ddp", help="Specify which gpu to train on. 'ddp' for parallel training, or '0', '1', etc.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    
    world_size = torch.cuda.device_count()
    if args.gpu=="ddp":
        mp.spawn(train, args=(world_size, args.config_path, args.data_folder, args.modelling_folder, True), nprocs=world_size, join=True)
    else:
        train(rank=int(args.gpu), world_size=world_size, config_path=args.config_path, data_folder=args.data_folder, modelling_folder=args.modelling_folder, ddp=False)