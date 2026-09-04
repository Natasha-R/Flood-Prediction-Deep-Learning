import numpy as np
import torch
import skimage
import cv2
import modelling.data_pipeline as data_pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tf
import os

def save_grad(x, gradients):
     x.register_hook(lambda z: gradients.append(z))

def seg_grad_cam(config, config_name, data_folder, modelling_folder, model, device, method,
                 patch, target_map, target_pred, box_top_left, box_bottom_right, class_of_interest):

    if box_top_left == box_bottom_right:
        box_bottom_right = [value+1 for value in box_bottom_right]

    dataset = data_pipeline.FloodDataset(config, data_folder, patch=patch)
    input_image = dataset[0]
    for item in input_image.keys():
        if item != "metadata":
            input_image[item] = input_image[item].unsqueeze(0).to(device)

    if config["architecture"].lower() == "basicunet":
        target_layer = model.up_convs[0].conv1
    elif target_map.lower() == "basin_up":
        target_layer = model.basin_up_convs[0].conv1
    elif target_map.lower() == "context_up":
        target_layer = model.context_up_convs[0].conv1
    elif target_map.lower() == "nearby_up":
        target_layer = model.nearby_up_convs[0].conv1
    elif target_map.lower() == "local_up":
        target_layer = model.local_up_convs[0].conv1
    elif target_map.lower() == "local_attend_higher":
        target_layer = model.local_attends_higher.conv
    elif target_map.lower() == "basin_down":
        target_layer = model.basin_down_convs[1].conv2
    elif target_map.lower() == "context_down":
        target_layer = model.context_down_convs[1].conv2
    elif target_map.lower() == "nearby_down":
        target_layer = model.nearby_down_convs[1].conv2
    elif target_map.lower() == "local_down":
        target_layer = model.local_down_convs[1].conv2
    elif target_map.lower() == "local_down_first":
        target_layer = model.local_down_convs[0].conv1
    else:
        raise ValueError(f"Unrecognised feature map: '{target_map}'")
        
    activations, gradients = [], []
    handle_1 = target_layer.register_forward_hook(lambda x, y, z: activations.append(z))
    handle_2 = target_layer.register_forward_hook(lambda x, y, z: save_grad(z, gradients))

    model.zero_grad()
    model.eval()
    output = model(input_image)[f"{target_pred}_pred"]
    predicted_class = torch.argmax(output, dim=1).squeeze().detach().cpu().numpy()
    label = input_image[f"{target_pred}_label"].squeeze().detach().cpu().numpy()

    mask = output[0].argmax(0).detach().cpu().numpy()
    mask_uint8 = 255 * np.uint8(mask == class_of_interest)
    mask_float = np.float32(mask == class_of_interest)
    mask_mask = np.zeros(mask_float.shape)
    mask_mask[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]] = 1
    mask_float = mask_float * mask_mask
    mask_uint8 = np.uint8(mask_uint8 * mask_mask)
    loss = (output[0, class_of_interest, :, :] * torch.tensor(mask_float).to(device)).sum()
    loss.backward()
    activations = activations[0][0].detach().cpu().numpy()
    gradients = gradients[0][0].detach().cpu().numpy()

    if method == "seg_xres_cam":
        pooled = skimage.measure.block_reduce(gradients, (1, 2, 2), np.max)
        pooled = np.transpose(pooled, (1, 2, 0))
        gradients = skimage.transform.resize(pooled, (gradients.shape[1], gradients.shape[2]), order = 0)
        gradients = np.transpose(gradients, (2, 0, 1))
        grayscale_cam = gradients * activations
        grayscale_cam = grayscale_cam.sum(axis = 0)
    else:
        coef = gradients.sum(axis = (1, 2))
        coef = coef[:, None, None]
        grayscale_cam = coef * activations
        grayscale_cam = grayscale_cam.sum(axis = 0)

    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = cv2.resize(grayscale_cam, (256, 256))
    max_, min_ = grayscale_cam.max(), grayscale_cam.min() 
    grayscale_cam = np.uint8(255 * (grayscale_cam - min_) / (max_ - min_))
    grayscale_cam = grayscale_cam / 255.0

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for ax in axes.ravel():
        for spine in ax.spines.values():
                spine.set_color('none')
        ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    predicted_class = torch.argmax(output, dim=1).squeeze().detach().cpu().numpy()

    correctly_matching = np.zeros((256, 256), dtype=np.uint8)
    correctly_matching[(label == 2) & (predicted_class == 2)] = 3 # flood predict correct
    correctly_matching[(label != 2) & (predicted_class == 2)] = 2 # overpredict flood
    correctly_matching[(label == 2) & (predicted_class != 2)] = 1 # underpredict flood
    matching_colours = {0: (0, 0, 0), 1:(255, 251, 0), 2:(202, 61, 23), 3:(35, 220, 71)} 
    pred_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for value, colour in matching_colours.items():
        pred_image[correctly_matching == value] = colour

    # colour_map = {0: (255, 0, 0), 1: (0, 0, 0), 2: (27, 197, 214), 3: (22, 130, 184), 4:(255, 251, 0), 5:(124, 252, 0)}
    # pred_image = np.zeros((256, 256, 3), dtype=np.uint8)
    # for value, colour in colour_map.items():
    #     pred_image[predicted_class == value] = colour

    axes[0].imshow(pred_image, interpolation="none")
    if not (box_top_left==(0, 0) and box_bottom_right==(256, 256)):
        axes[0].add_patch(patches.Rectangle((box_top_left[0], box_top_left[1]), box_bottom_right[0]-box_top_left[0], box_bottom_right[1]-box_top_left[1], 
                                            linewidth=5, edgecolor="white", facecolor="none"))
    class_name = {1:"\'no flood\'", 2:"\'flood\'"}[class_of_interest]
    axes[0].set_title(f"Model prediction of {target_pred} patch\n(Explaining the {class_name} class in the white square region)")

    target_map_folder = target_map.split("_")[0]
    target_map_folder = "con_context" if target_map_folder == "context" and config["use_consistent_context"] else target_map_folder
    s2_image = tf.imread(f"{os.environ['DATA_FOLDER']}/{target_map_folder}/sentinel2/{patch}")[:, :, (2, 1, 0)]
    #s2_image = (s2_image - np.min(s2_image)) / (np.max(s2_image) - np.min(s2_image))
    s2_image = np.clip((s2_image - np.quantile(s2_image, 0.01)) / (np.quantile(s2_image, 0.99) - np.quantile(s2_image, 0.01)), 0, 1)
    axes[1].imshow(s2_image)
    if target_map_folder != "local":
        top_left = (115, 115) if target_map_folder == "context" or target_map_folder == "con_context" else (77, 77)
        bottom_right = (141, 141) if target_map_folder == "context" or target_map_folder == "con_context" else (179, 179)
        axes[1].add_patch(patches.Rectangle((top_left[0], top_left[1]), bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], linewidth=3, edgecolor="white", facecolor="none"))
    class_name = {1:"\'no flood\'", 2:"\'flood\'"}[class_of_interest]

    grayscale_cam = np.ma.masked_where(grayscale_cam == 0, grayscale_cam)
    axes[1].imshow(grayscale_cam, alpha=0.7, cmap="jet", interpolation="bilinear")
    axes[1].set_title(f"Explanation by {method}\nfrom {target_map} feature map")

    fig.tight_layout()
    save_file = f"{config_name}_{method}_PRED{target_pred}_{patch}_{box_top_left}{box_bottom_right}_class{class_of_interest}_MAP{target_map}"
    fig.savefig(f"{modelling_folder}/visualise/{config_name}/{save_file}.png", bbox_inches="tight")
    print("Saved figure to:", f"{modelling_folder}/visualise/{config_name}/{save_file}.png", flush=True)