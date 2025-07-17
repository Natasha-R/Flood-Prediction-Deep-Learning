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
from visualise import plot_losses

def train(config_path, data_folder, modelling_folder, device, logger):

    config, config_name = utils.load_config(config_path, logger)
    model = utils.load_model(config, device, logger)
    train_loader, validation_loader = data_pipeline.create_data_loaders(config=config, data_folder=data_folder)
    loss_function = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0, size_average=True).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=config["learning_rate"], total_steps=config["number_epochs"])

    losses = {"total_train_losses": [], "total_validation_losses": [],
              "epoch_train_losses": [], "epoch_validation_losses": []}

    for epoch in range(1, config["number_epochs"]+1):

        epoch_start_time = time.time()
        losses["epoch_train_losses"], losses["epoch_validation_losses"] = [], []

        model.train()
        for data in tqdm(train_loader, desc="Training:", leave=False):
            optimizer.zero_grad()
            local_features = data["local_features"] # BCHW
            local_label = data["local_label"].to(device) # BHW
            model_output = model(local_features) #BclassesHW
            loss = loss_function(model_output, local_label)
            loss.backward()
            optimizer.step()
            losses["epoch_train_losses"].append(loss.item())
            
        model.eval()
        with torch.no_grad():
            for data in tqdm(validation_loader, desc="Validation loss:", leave=False):
                local_features = data["local_features"]
                local_label = data["local_label"].to(device)
                model_output = model(local_features)
                loss = loss_function(model_output, local_label)
                losses["epoch_validation_losses"].append(loss.item())

        scheduler.step()

        losses["total_train_losses"].append(sum(losses["epoch_train_losses"]) / len(losses["epoch_train_losses"]))
        losses["total_validation_losses"].append(sum(losses["epoch_validation_losses"]) / len(losses["epoch_validation_losses"]))
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} ({epoch_duration:.0f} seconds) | Train loss: {losses["total_train_losses"][-1]:.3f} | Validation loss: {losses["total_validation_losses"][-1]:.3f}")

        if config["save_model_on_epoch"] != 0 and ((epoch % config["save_model_on_epoch"]) == 0):
            model_save_path = f"{modelling_folder}/models/{config_name}_{epoch}.pth"
            torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), model_save_path)
            logger.info(f"Saved the model to: {model_save_path}")

        if config["calculate_metrics_on_epoch"] != 0 and ((epoch % config["calculate_metrics_on_epoch"]) == 0):
            metrics.calculate_metrics(config, config_name, model, validation_loader, modelling_folder, epoch, logger, device, subset="validation")

    plot_losses(losses["total_train_losses"], losses["total_validation_losses"], config_name, modelling_folder)

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