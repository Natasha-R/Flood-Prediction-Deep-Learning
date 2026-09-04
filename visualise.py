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
from rasterio.windows import from_bounds
import json

def hatch_mask(mask, spacing=8, thickness=1, direction="+"):
    yy, xx = np.indices(mask.shape)
    if direction == "+":
        pattern = ((xx + yy) % spacing) < thickness
    else:
        pattern = ((xx - yy) % spacing) < thickness
    hatch = mask & pattern
    return hatch

def denormalise_feature(patch, predict_feature, data_folder):

    with open(f"{data_folder}/metadata/zscore.json") as file:
        zscore = json.load(file)
    patch = (patch * zscore[predict_feature][str(0)]["scale"]) + zscore[predict_feature][str(0)]["shift"]
    if predict_feature == "dem":
        patch = torch.exp(patch) - 200
    elif predict_feature == "sentinel2":
        for band in [0, 1, 2, 9]:
            patch[band] = torch.exp(patch[band])
        patch -= 2
    elif predict_feature in ["precipitation", "summary_precipitation", "slope", "hand"]:
        patch = torch.exp(patch) - 1
    return patch

def visualise_predictions(config, config_name, model, num_epochs, data_folder, modelling_folder, device, subevent, file_type, 
                          scale, patch, pred_only, border, class_probabilities=False, mask_features=None, mask_before_normalization=False, mask_value=0):

    loss_function = "cross entropy"
    if config.get("loss_function", "cross entropy").lower()=="dice":
        loss_function = "dice"
    predict_feature = config.get("predict_feature", False)
    torch_dtype = torch.int8 if not predict_feature else torch.float32
    numpy_dtype = "int8" if not predict_feature else "float32"

    # load the dataset
    dataset = data_pipeline.FloodDataset(config, data_folder, subevent=subevent, patch=patch, mask_features=mask_features, mask_before_normalization=mask_before_normalization, mask_value=mask_value)
    if patch: subevent=patch

    if mask_features:
        base_config = "_".join([element for element in config_name.split("_") if not("[" in element)])
        path_folder = f"{modelling_folder}/visualise/{base_config}/"
    else:
        path_folder = f"{modelling_folder}/visualise/{config_name}/"
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)

    # define the colours for the visualisation
    # 0: no data, 1: aoi, 2: flood, 4: val border, 5: test border, 6: train border
    label_colours = {0: (255, 0, 0), 1: (0, 0, 0), 2: (27, 197, 214), 4:(255, 255, 255), 5:(0, 0, 0), 6: (152, 97, 255)} 
    # 0 no data, 1: underpredict, 2: overpredict, 3: correct, 4: val border, 5: test border, 6: train border
    matching_colours = {0: (0, 0, 0), 1:(255, 251, 0), 2:(202, 61, 23), 3:(35, 220, 71), 4:(255, 255, 255), 5:(0, 0, 0), 6: (152, 97, 255)} 
    all_patch_paths, all_prob_patch_paths = [], []

    model.eval()
    with torch.no_grad():

        for patch_index in range(len(dataset)):

            # load in the data patch from the dataset
            patch_file = dataset.patches[patch_index]
            sample = dataset[patch_index]
            for item in sample.keys():
                if "metadata" not in item:
                    sample[item] = sample[item].unsqueeze(0).to(device)

            # make a model prediction from the patch
            model_output = model(sample)
            if loss_function=="dice":
                predicted_class = model_output[f"{scale}_pred"]
                predicted_class = (torch.sigmoid(predicted_class.squeeze()) > 0.5) * 1
                label = sample[f"{scale}_label"].squeeze()
            elif not predict_feature:
                predicted_class = torch.argmax(model_output[f"{scale}_pred"], dim=1).squeeze()
                label = sample[f"{scale}_label"].squeeze()
            else: 
                predicted_class = denormalise_feature(model_output[f"{scale}_pred"].squeeze(), predict_feature, data_folder)
                label = denormalise_feature(sample[f"{scale}_label"].squeeze(), predict_feature, data_folder)
                sample[f"{scale}_label"] = label
            if "resnet" in config["architecture"].lower():
                predicted_class = torch.ones(256, 256).to(device) * predicted_class
                label = torch.ones(256, 256).to(device) * label
                sample[f"{scale}_label"] = label
            # if classification:
            #     predicted_class, label = convert_to_classification(predicted_class.unsqueeze(0), label.unsqueeze(0), config)
            #     predicted_class, label = predicted_class.squeeze(), label.squeeze()
            #     label[label == 1] = 2
            #     label[label == 0] = 1
            if loss_function=="dice":
                predicted_class[predicted_class == 1] = 2
                predicted_class[predicted_class == 0] = 1
            if class_probabilities:
                predicted_probabilities = torch.softmax(model_output[f"{scale}_pred"], dim=1)
                predicted_class_probability = predicted_probabilities.gather(1, torch.argmax(predicted_probabilities, dim=1).unsqueeze(1)).squeeze(1)
                predicted_probabilities = torch.concat([predicted_probabilities.squeeze(), predicted_class_probability]).squeeze()

            # evaluate the matching of prediction to label
            if not predict_feature:
                correctly_matching = torch.zeros_like(label)
                correctly_matching[(label == 2) & (predicted_class == 2)] = 3 # flood predict correct
                correctly_matching[(label != 2) & (predicted_class == 2)] = 2 # overpredict flood
                correctly_matching[(label == 2) & (predicted_class != 2)] = 1 # underpredict flood
            else:
                correctly_matching = torch.zeros_like(label)
            
            # use the corresponding label file as a reference for the size and geographical bounds of the data patch
            with rasterio.open(f"{data_folder}/{scale}/label/{patch_file}") as reference_file:
                meta = reference_file.meta.copy()
                meta.update({"count": 3, "dtype": numpy_dtype, "nodata":0})

            # save the predictions for the data patch
            patch_path = f"{path_folder}/{config_name}_{patch_file}"
            with rasterio.open(patch_path, "w", **meta, compress="LZW") as file:
                file.write(predicted_class.squeeze().to(torch_dtype).cpu().numpy(), 1)
                file.write(sample[f"{scale}_label"].squeeze().to(torch_dtype).cpu().numpy(), 2)
                file.write(correctly_matching.to(torch_dtype).cpu().numpy(), 3)
                file.nodata = 0
            all_patch_paths.append(patch_path)

            if class_probabilities:
                prob_patch_path = f"{path_folder}/{config_name}_prob_{patch_file}"
                meta.update({"count": 4, "dtype": "float32"})
                meta.pop("nodata", None)
                with rasterio.open(prob_patch_path, "w", **meta, compress="LZW") as file:
                    file.write(predicted_probabilities.cpu().numpy())
                all_prob_patch_paths.append(prob_patch_path)

    # merge all of the predicted patches together
    all_patches = [rasterio.open(path) for path in all_patch_paths]
    full_subevent, full_subevent_transform = merge(all_patches)
    if not predict_feature:
        for index in [0, 2]: full_subevent[index] = np.where(full_subevent[1] == 0, 0, full_subevent[index])
    full_subevent_meta = all_patches[0].meta.copy()
    full_subevent_meta.update({"height": full_subevent.shape[1], "width": full_subevent.shape[2],
                               "transform": full_subevent_transform, "dtype": numpy_dtype, "nodata": 0})

    if class_probabilities:
        all_prob_patches = [rasterio.open(path) for path in all_prob_patch_paths]
        full_prob_subevent, full_prob_subevent_transform = merge(all_prob_patches)
        #for index in [0, 2]: full_subevent[index] = np.where(full_subevent[1] == 0, 0, full_subevent[index])
        full_prob_subevent_meta = all_prob_patches[0].meta.copy()
        full_prob_subevent_meta.update({"height": full_prob_subevent.shape[1], "width": full_prob_subevent.shape[2], "transform": full_prob_subevent_transform})

    # mark a border around the patches
    if border > 0:
        patch_subset = list(dataset.data_subset["subset"])

        for index, subset_name in enumerate(patch_subset):
            if "val" in subset_name:
                value = 4
            elif "test" in subset_name:
                value = 5
            elif "train" in subset_name:
                value = 6
            else:
                continue
            window = from_bounds(*all_patches[index].bounds, transform=full_subevent_transform)
            window = window.round_offsets().round_lengths()
            row_start = max(0, int(window.row_off))
            col_start = max(0, int(window.col_off))
            row_end = min(full_subevent.shape[1], row_start + int(window.height))
            col_end = min(full_subevent.shape[2], col_start + int(window.width))

            patch_view = full_subevent[:, row_start:row_end, col_start:col_end]
            height = patch_view.shape[1]
            width = patch_view.shape[2]
            if height == 0 or width == 0:
                continue

            if "train" in subset_name:
                mask = np.ones((height, width), dtype=bool)
                hatch = hatch_mask(mask, spacing=50, thickness=border, direction="+")
                hatch |= hatch_mask(mask, spacing=50, thickness=border, direction="-")
                for band in range(3):
                    patch_view[band][hatch] = value

            thickness = min(border, height, width)
            patch_view[:, :thickness, :] = value
            patch_view[:, -thickness:, :] = value
            patch_view[:, :, :thickness] = value
            patch_view[:, :, -thickness:] = value

    # save the visualisation as geotiff, with predicted, ground truth label, and matching bands
    save_path = f"{path_folder}/{config_name}_{num_epochs}epochs_{scale}_{subevent}"
    if mask_features:
         save_path = f"{save_path}_value{mask_value}_beforenorm{mask_before_normalization}"

    if file_type == "geotiff":
        with rasterio.open(f"{save_path}.tif", "w", **full_subevent_meta, compress="LZW") as file:
            file.write(full_subevent)
            for band_name, band_num, colour_map in zip(["Model Predicted Class", "Label Class", "Correctly Matching"], [1, 2, 3], 
                                                       [label_colours, label_colours, matching_colours]):
                file.set_band_description(band_num, band_name)
                file.write_colormap(band_num, colour_map)
            file.nodata = 0

        if class_probabilities:
            with rasterio.open(f"{save_path}_prob.tif", "w", **full_prob_subevent_meta, compress="LZW") as file:
                file.write(full_prob_subevent)
                file.set_band_description(1, "no_data_probability")
                file.set_band_description(2, "no_flood_probability")
                file.set_band_description(3, "flood_probability")
                file.set_band_description(4, "predicted_class_probability")

    # save the visualisation as a jpeg, with separate files for the predictions, ground truth label, and matching
    elif file_type == "png":
        image_names, indices, colour_maps = ["prediction", "label", "match"], [0, 1, 2], [label_colours, label_colours, matching_colours]
        if pred_only:
            image_names, indices, colour_maps = [image_names[0]], [indices[0]], [colour_maps[0]]
        for image_name, index, colour_map in zip(image_names, indices, colour_maps):
            rgb_image = np.zeros((full_subevent.shape[1], full_subevent.shape[2], 3), dtype=numpy_dtype)
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
    if class_probabilities:
        for file in all_prob_patches:
            file.close()
        for path in all_prob_patch_paths:
            if os.path.exists(path):
                os.remove(path)

    print(f"Saved visualisation to {save_path}")

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
    parser.add_argument('--class_probabilities', action="store_true", default=False, help="Also generate a visualisation of the class probabilities.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    args.gpu = int(args.gpu) if args.gpu != "cpu" and args.gpu != "ddp" else args.gpu

    config, config_name = utils.load_config(args.config_path)
    num_epochs = args.epochs if args.epochs else config['number_epochs']
    model = utils.load_model(config, rank=args.gpu, ddp=False, pretrained_path=f"{args.modelling_folder}/models/{config_name}_{num_epochs}.pth")

    visualise_predictions(config=config, config_name=config_name, model=model, num_epochs=num_epochs, data_folder=args.data_folder, modelling_folder=args.modelling_folder, 
                          device=args.gpu, subevent=args.subevent, file_type=args.file_type, scale=args.scale, patch=args.patch, 
                          pred_only=args.pred_only, border=int(args.border), class_probabilities=args.class_probabilities)