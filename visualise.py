import modelling.utils as utils
import torch
import argparse
import modelling.data_pipeline as data_pipeline
import rasterio
from rasterio.merge import merge
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualise_predictions(config_path, data_folder, modelling_folder, device, logger, subevent, file_type):

    # load the config, model, and dataset
    config, config_name = utils.load_config(config_path, logger)
    model = utils.load_model(config, device, logger, pretrained_path=f"{modelling_folder}/models/{config_name}_{config['number_epochs']}.pth")
    dataset = data_pipeline.FloodDataset(config, data_folder, subevent=subevent)

    # define the colours for the visualisation
    label_colours = {0: (255, 0, 0), 1: (0, 0, 0), 2: (27, 197, 214), 3: (22, 130, 184)}
    matching_colours = {0: (0, 0, 0), 1: (202, 61, 23), 2: (35, 220, 71)}
    all_patch_paths = []

    for patch_index in range(len(dataset)):

        # load in the data patch from the dataset
        patch_file = dataset.patches[patch_index]
        local_features = dataset[patch_index]["local_features"]
        for feature in local_features: local_features[feature] = local_features[feature].unsqueeze(0)
        local_label = dataset[patch_index]["local_label"].to(device)

        # make a model prediction from the patch
        model_output = model(local_features)
        predicted_class = torch.argmax(model_output, dim=1)
        correctly_matching = ((local_label==predicted_class.squeeze())*1) + 1 # 0 is no data, 1 is incorrect, 2 is correct
        
        # use the corresponding label file as a reference for the size and geographical bounds of the data patch
        with rasterio.open(f"{data_folder}/local/label/{patch_file}") as reference_file:
            meta = reference_file.meta.copy()
            meta.update({"count": 3, "dtype": "int8", "nodata":0})

        # save the predictions for the data patch
        patch_path = f"{modelling_folder}/visualise/{config_name}_{patch_file}"
        with rasterio.open(patch_path, "w", **meta, compress="LZW") as file:
            file.write(predicted_class.squeeze().to(torch.int8).cpu().numpy(), 1)
            file.write(local_label.to(torch.int8).cpu().numpy(), 2)
            file.write(correctly_matching.to(torch.int8).cpu().numpy(), 3)
            file.nodata = 0

        all_patch_paths.append(patch_path)

    # merge all of the predicted patches together
    all_patches = [rasterio.open(path) for path in all_patch_paths]
    full_subevent, full_subevent_transform = merge(all_patches)
    for index in [0, 2]: full_subevent[index] = np.where(full_subevent[1] == 0, 0, full_subevent[index])
    full_subevent_meta = all_patches[0].meta.copy()
    full_subevent_meta.update({"height": full_subevent.shape[1], "width": full_subevent.shape[2],
                               "transform": full_subevent_transform, "dtype": "int8", "nodata": 0})
    
    # save the visualisation as geotiff, with predicted, ground truth label, and matching bands
    if file_type == "geotiff":
        with rasterio.open(f"{modelling_folder}/visualise/{config_name}_{subevent}.tif", "w", **full_subevent_meta, compress="LZW") as file:
            file.write(full_subevent)
            for band_name, band_num, colour_map in zip(["Model Predicted Class", "Label Class", "Correctly Matching"], [1, 2, 3], 
                                                       [label_colours, label_colours, matching_colours]):
                file.set_band_description(band_num, band_name)
                file.write_colormap(band_num, colour_map)
            file.nodata = 0

    # save the visualisation as a jpeg, with separate files for the predictions, ground truth label, and matching
    else:
        for image_name, index, colour_map in zip(["prediction", "label", "match"], [0, 1, 2], [label_colours, label_colours, matching_colours]):
            rgb_image = np.zeros((full_subevent.shape[1], full_subevent.shape[2], 3), dtype=np.uint8)
            for value, colour in colour_map.items():
                for channel in range(3):
                    rgb_image[..., channel][full_subevent[index] == value] = colour[channel]
            Image.fromarray(rgb_image).save(f"{modelling_folder}/visualise/{config_name}_{subevent}_{image_name}.jpg", 'JPEG')

    # close all open files and delete all temporary files
    for file in all_patches:
        file.close()
    for path in all_patch_paths:
        os.remove(path)

    logger.info(f"Saved visualisation of '{subevent}' by model '{config_name}'.")

def plot_losses(train_loss, validation_loss, config_name, modelling_folder):
    """
    Plot the train and validation losses from model training.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax.plot(range(1, len(train_loss)+1), train_loss, c="black", label="Train loss", linewidth=2)
    ax.plot(range(1, len(validation_loss)+1), validation_loss, c="blue", label="Validation loss", linewidth=2)
    ax.set_title(f"Train and validation losses", fontsize=15)
    ax.legend(fontsize=14), ax.grid(alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("Epoch", fontsize=14), ax.set_ylabel("Loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{modelling_folder}/losses/{config_name}.png", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the model's predictions.")
    parser.add_argument('--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('--gpu', default="cuda", help="Specify which gpu to use. 'cuda', 'cuda:0', 'cuda:1', etc.")

    parser.add_argument('--subevent', default=None, help="Visualise model predictions of the specified subevent.")
    parser.add_argument('--file_type', default="geotiff", help="Save the image file as either a 'geotiff' or 'jpg'.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    logger = utils.get_logger()

    visualise_predictions(config_path=args.config_path, data_folder=args.data_folder, modelling_folder=args.modelling_folder, 
                          device=args.gpu, logger=logger, subevent=args.subevent, file_type=args.file_type)