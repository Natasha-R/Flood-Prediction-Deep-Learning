import modelling.data_pipeline as data_pipeline
import metrics
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import os 
import modelling.utils as utils
import argparse
import pandas as pd
from visualise import plot_losses
import modelling.architectures as architectures

def train(config_path, data_folder, modelling_folder, device, logger):

    config, config_name = utils.load_config(config_path, logger)
    model = utils.load_model(config, device, logger)
    subsets = list(pd.read_csv(f"{data_folder}/metadata/{config['data_subset_file']}.csv")["subset"].unique())
    data_loaders = {subset: data_pipeline.create_data_loader(config, data_folder, subset) for subset in subsets}
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0, size_average=True)
    #loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=config["learning_rate"], total_steps=config["number_epochs"])

    losses = {f"total_{subset}_losses": [] for subset in subsets}
    losses.update({f"epoch_{subset}_losses": [] for subset in subsets})

    for epoch in range(1, config["number_epochs"]+1):

        epoch_start_time = time.time()
        losses.update({f"epoch_{subset}_losses": [] for subset in subsets})

        model.train()
        for data in tqdm(data_loaders["train"], desc="Training", leave=False):
            optimizer.zero_grad()
            for item in data.keys():
                data[item] = data[item].to(device) #BCHW (features) #BHW (label)
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
                        data[item] = data[item].to(device) #BCHW (features) #BHW (label)
                    model_output = model(data)
                    loss = 0.0
                    for scale in config["scales"]:
                        loss = loss + config[f"{scale}_weight"] * loss_function(model_output[f"{scale}_pred"], data[f"{scale}_label"])
                    losses[f"epoch_{validation_loader}_losses"].append(loss.item())

        scheduler.step()

        loss_string = f"Epoch {epoch} ({time.time() - epoch_start_time:.0f} seconds)"
        for loss_type in subsets:
            losses[f"total_{loss_type}_losses"].append(sum(losses[f"epoch_{loss_type}_losses"]) / len(losses[f"epoch_{loss_type}_losses"]))
            loss_string += f" | {loss_type} loss: {losses[f'total_{loss_type}_losses'][-1]:.3f}"
        logger.info(loss_string)

        if config["save_model_on_epoch"] != 0 and ((epoch % config["save_model_on_epoch"]) == 0):
            model_save_path = f"{modelling_folder}/models/{config_name}_{epoch}.pth"
            torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), model_save_path)
            logger.info(f"Saved the model to: {model_save_path}")

        if config["calculate_metrics_on_epoch"] != 0 and ((epoch % config["calculate_metrics_on_epoch"]) == 0):
            for type in subsets:
                metrics.calculate_metrics(config, config_name, model, data_loaders[type], modelling_folder, epoch, logger, device, subset=type)

    plot_losses(losses, config_name, modelling_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('--gpu', default="cuda", help="Specify which gpu to train on. 'cuda' for parallel training, or 'cuda:0', 'cuda:1', etc.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    logger = utils.get_logger()

    train(config_path=args.config_path, data_folder=args.data_folder, modelling_folder=args.modelling_folder, device=args.gpu, logger=logger)