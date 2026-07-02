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
from rasterio.features import rasterize
from scipy.ndimage import binary_erosion
from shapely.geometry import box
import geopandas as gpd
from metrics import convert_to_classification

def hatch_mask(mask, spacing=8, thickness=1, direction="+"):
    yy, xx = np.indices(mask.shape)
    if direction == "+":
        pattern = ((xx + yy) % spacing) < thickness
    else:
        pattern = ((xx - yy) % spacing) < thickness
    hatch = mask & pattern
    return hatch

def visualise_predictions(config, config_name, model, num_epochs, data_folder, modelling_folder, device, subevent, file_type, 
                          scale, patch, pred_only, border, classification, sensitivity, resolution, mask_features=None, mask_patch=None):

    loss_function = "cross entropy"
    if config.get("loss_function", "cross entropy").lower()=="dice":
        loss_function = "dice"
    if classification:
        config["classification_evaluation"] = classification
        config["sensitivity"] = sensitivity
        config["resolution"] = resolution

    # load the dataset
    dataset = data_pipeline.FloodDataset(config, data_folder, subevent=subevent, patch=patch, mask_features=mask_features, mask_patch=mask_patch)
    if patch: subevent=patch

    path_folder = f"{modelling_folder}/visualise/{config_name}/"
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)

    # define the colours for the visualisation
    # 0: no data, 1: aoi, 2: flood, 4: val border, 5: test border, 6: train border
    label_colours = {0: (255, 0, 0), 1: (0, 0, 0), 2: (27, 197, 214), 4:(255, 255, 255), 5:(152, 97, 255), 6: (152, 97, 255)} 
    # 0 no data, 1: underpredict, 2: overpredict, 3: correct, 4: val border, 5: test border, 6: train border
    matching_colours = {0: (0, 0, 0), 1:(255, 251, 0), 2:(202, 61, 23), 3:(35, 220, 71), 4:(255, 255, 255), 5:(152, 97, 255), 6: (152, 97, 255)} 
    all_patch_paths = []

    model.eval()
    with torch.no_grad():

        for patch_index in range(len(dataset)):

            # load in the data patch from the dataset
            patch_file = dataset.patches[patch_index]
            sample = dataset[patch_index]
            for item in sample.keys():
                sample[item] = sample[item].unsqueeze(0).to(device)

            # make a model prediction from the patch
            model_output = model(sample)
            if loss_function=="dice":
                predicted_class = model_output[f"{scale}_pred"]
                predicted_class = (torch.sigmoid(predicted_class.squeeze()) > 0.5) * 1
            else:
                predicted_class = torch.argmax(model_output[f"{scale}_pred"], dim=1).squeeze()
            label = sample[f"{scale}_label"].squeeze()
            if "resnet" in config["architecture"].lower():
                predicted_class = torch.ones(256, 256).to(device) * predicted_class
                label = torch.ones(256, 256).to(device) * label
                sample[f"{scale}_label"] = label
            if classification:
                predicted_class, label = convert_to_classification(predicted_class.unsqueeze(0), label.unsqueeze(0), config)
                predicted_class, label = predicted_class.squeeze(), label.squeeze()
                label[label == 1] = 2
                label[label == 0] = 1
            if loss_function=="dice" or classification:
                predicted_class[predicted_class == 1] = 2
                predicted_class[predicted_class == 0] = 1

            # evaluate the matching of prediction to label
            correctly_matching = torch.zeros_like(label)
            correctly_matching[(label == 2) & (predicted_class == 2)] = 3 # flood predict correct
            correctly_matching[(label != 2) & (predicted_class == 2)] = 2 # overpredict flood
            correctly_matching[(label == 2) & (predicted_class != 2)] = 1 # underpredict flood
            
            # use the corresponding label file as a reference for the size and geographical bounds of the data patch
            with rasterio.open(f"{data_folder}/{scale}/label/{patch_file}") as reference_file:
                meta = reference_file.meta.copy()
                meta.update({"count": 3, "dtype": "int8", "nodata":0})

            # save the predictions for the data patch
            patch_path = f"{path_folder}/{config_name}_{patch_file}"
            with rasterio.open(patch_path, "w", **meta, compress="LZW") as file:
                file.write(predicted_class.squeeze().to(torch.int8).cpu().numpy(), 1)
                file.write(sample[f"{scale}_label"].squeeze().to(torch.int8).cpu().numpy(), 2)
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

    # mark a border around the patches
    if border > 0:
        patch_subset = list(dataset.data_subset["subset"])
        for other_subset, value in zip(["val", "test", "train"], [4, 5, 6]):
            patch_geometries = gpd.GeoDataFrame(geometry=[box(*all_patches[index].bounds) for index in range(len(dataset)) if other_subset in patch_subset[index]])
            if len(patch_geometries) > 0:
                # patch_geometries = patch_geometries.buffer(0.0001).union_all().buffer(-0.0001)
                # patch_geometries = patch_geometries.geoms if patch_geometries.geom_type == "MultiPolygon" else [patch_geometries]
                for geometry in patch_geometries.geometry:
                    mask = rasterize([(geometry, 1)], out_shape=(full_subevent.shape[1], full_subevent.shape[2]), transform=full_subevent_transform, fill=0, dtype=np.int8).astype(bool)
                    if other_subset == "train":
                        hatch = hatch_mask(mask, spacing=50, thickness=border, direction="+")
                        hatch |= hatch_mask(mask, spacing=50, thickness=border, direction="-")
                        for band in range(3):
                            full_subevent[band][hatch] = value
                    border_mask = (mask) & (~binary_erosion(mask, iterations=border))
                    for band in range(3):
                        full_subevent[band][border_mask] = value

    # save the visualisation as geotiff, with predicted, ground truth label, and matching bands
    save_path = f"{path_folder}/{config_name}_{num_epochs}epochs_{scale}_{subevent}"
    if classification:
        save_path = f"{save_path}_sen{str(sensitivity).replace(".","-")}_res{resolution}"

    if file_type == "geotiff":
        with rasterio.open(f"{save_path}.tif", "w", **full_subevent_meta, compress="LZW") as file:
            file.write(full_subevent)
            for band_name, band_num, colour_map in zip(["Model Predicted Class", "Label Class", "Correctly Matching"], [1, 2, 3], 
                                                       [label_colours, label_colours, matching_colours]):
                file.set_band_description(band_num, band_name)
                file.write_colormap(band_num, colour_map)
            file.nodata = 0

    # save the visualisation as a jpeg, with separate files for the predictions, ground truth label, and matching
    elif file_type == "png":
        image_names, indices, colour_maps = ["prediction", "label", "match"], [0, 1, 2], [label_colours, label_colours, matching_colours]
        if pred_only:
            image_names, indices, colour_maps = [image_names[0]], [indices[0]], [colour_maps[0]]
        for image_name, index, colour_map in zip(image_names, indices, colour_maps):
            rgb_image = np.zeros((full_subevent.shape[1], full_subevent.shape[2], 3), dtype=np.uint8)
            for value, colour in colour_map.items():
                for channel in range(3):
                    rgb_image[..., channel][full_subevent[index] == value] = colour[channel]
            Image.fromarray(rgb_image).save(f"{save_path}_{image_name}.png", 'PNG')

    # close all open files and delete all temporary files
    for file in all_patches:
        file.close()
    for path in all_patch_paths:
        if os.path.exists(path):
            os.remove(path)

    print(f"Saved visualisation of '{subevent}' by model '{config_name}'.")

def plot_losses(losses, config, config_name, modelling_folder, rank, logger):
    """
    Plot the train and validation losses from model training.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    loss_types = ["total_train_losses"] + [loss_type for loss_type in losses.keys() if "train" not in loss_type and "epoch" not in loss_type and "local" not in loss_type]
    colours = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#ffff33','#999999','#000000'][:len(loss_types)]
    
    for loss_type, colour in zip(loss_types, colours):
        local_loss_type = "_".join(loss_type.split("_")[:-1]) + "_local_losses"
        for loss_name, line in zip(*([loss_type, local_loss_type], ["-", "--"]) if not config["only_pred_local"] else ([local_loss_type], ["--"])):
            ax.plot(range(1, len(losses[loss_name])+1), losses[loss_name], line, c=colour, linewidth=2, label=" ".join(loss_name.split("_")[1:-1]).title() + " Loss")

    ax.set_title(f"Train and Validation Losses (GPU {rank})", fontsize=15)
    ax.legend(fontsize=14), ax.grid(alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("Epoch", fontsize=14), ax.set_ylabel("Loss", fontsize=14)
    
    fig.tight_layout()
    fig.savefig(f"{modelling_folder}/losses/{config_name}_gpu{rank}.png", bbox_inches="tight")
    logger.info(f"Saved loss plot to: {modelling_folder}/losses/{config_name}_gpu{rank}.png")

    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{modelling_folder}/losses/01_{config_name}_gpu{rank}.png", bbox_inches="tight")
    logger.info(f"Saved loss plot to: {modelling_folder}/losses/01_{config_name}_gpu{rank}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the model's predictions.")
    parser.add_argument('-c', '--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('-d', '--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('-m', '--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('-g', '--gpu', default="0", help="Specify which gpu to use. '0', '1', etc.")

    parser.add_argument('-e', '--epochs', default=None, help="Load the model trained to the specified number of epochs.")
    parser.add_argument('-s', '--subevent', default=None, help="Visualise model predictions of the specified subevent.")
    parser.add_argument('-p', '--patch', default=None, help="Visualise model predictions of the specified patch.")
    parser.add_argument('-a', '--scale', default="local", help="Define the scale for the visualisation.")
    parser.add_argument('-f', '--file_type', default="geotiff", help="Save the image file as either a 'geotiff' or 'png'.")
    parser.add_argument('-o', '--pred_only', action="store_true", default=False, help="Save only the prediction, and not label & comparison")
    parser.add_argument('-b', '--border', default=5, help="Print a border around the patches, with the given pixel size")

    parser.add_argument('--classification', action="store_true", default=False, help="Evaluate using a classification approach.")
    parser.add_argument('--sensitivity', type=float, default=0.05, help="Specify a sensitivity for the flood proportion.")
    parser.add_argument('--resolution', type=int, default=1, help="Specify the resolution for the classification evaluation.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    args.gpu = int(args.gpu) if args.gpu != "cpu" and args.gpu != "ddp" else args.gpu

    config, config_name = utils.load_config(args.config_path)
    num_epochs = args.epochs if args.epochs else config['number_epochs']
    model = utils.load_model(config, rank=args.gpu, ddp=False, pretrained_path=f"{args.modelling_folder}/models/{config_name}_{num_epochs}.pth")

    visualise_predictions(config=config, config_name=config_name, model=model, num_epochs=num_epochs, data_folder=args.data_folder, modelling_folder=args.modelling_folder, 
                          device=args.gpu, subevent=args.subevent, file_type=args.file_type, scale=args.scale, patch=args.patch, 
                          pred_only=args.pred_only, border=int(args.border), classification=args.classification, sensitivity=args.sensitivity, resolution=args.resolution)