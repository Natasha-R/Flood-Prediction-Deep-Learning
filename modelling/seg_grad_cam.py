import numpy as np
import torch
import skimage
import cv2
import modelling.data_pipeline as data_pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_grad(x, gradients):
     x.register_hook(lambda z: gradients.append(z))

def seg_grad_cam(config, config_name, data_folder, modelling_folder, model, device, method, logger,
                 patch, target_map, target_pred, box_top_left, box_bottom_right, class_of_interest):

    if box_top_left == box_bottom_right:
        box_bottom_right = [value+1 for value in box_bottom_right]

    dataset = data_pipeline.FloodDataset(config, data_folder, patch=patch)
    input_image = dataset[0]
    for item in input_image.keys():
        input_image[item] = input_image[item].unsqueeze(0).to(device)

    if config["architecture"].lower() == "basicunet":
        target_layer = model.up_convs[0].conv1
    elif target_map.lower() == "basin":
        target_layer = model.basin_up_convs[0].conv1
    elif target_map.lower() == "context":
        target_layer = model.context_up_convs[0].conv1
    elif target_map.lower() == "local":
        target_layer = model.local_up_convs[0].conv1
    elif target_map.lower() == "context_attend_basin":
        target_layer = model.context_attends_basin.conv
    elif target_map.lower() == "local_attend_higher":
        target_layer = model.local_attends_higher.conv

    activations, gradients = [], []
    handle_1 = target_layer.register_forward_hook(lambda x, y, z: activations.append(z))
    handle_2 = target_layer.register_forward_hook(lambda x, y, z: save_grad(z, gradients))

    model.zero_grad()
    model.eval()
    output = model(input_image)[f"{target_pred}_pred"]
    predicted_class = torch.argmax(output, dim=1).squeeze().detach().cpu().numpy()

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
    colour_map = {0: (255, 0, 0), 1: (0, 0, 0), 2: (27, 197, 214), 3: (22, 130, 184), 4:(255, 251, 0), 5:(124, 252, 0)}
    pred_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for value, colour in colour_map.items():
        pred_image[predicted_class == value] = colour
    axes[0].imshow(pred_image, interpolation="none")
    axes[0].add_patch(patches.Rectangle((box_top_left[1], box_top_left[0]), box_bottom_right[1]-box_top_left[1], box_bottom_right[0]-box_top_left[0], 
                                        linewidth=5, edgecolor="red", facecolor="none"))
    class_name = {1:"\'no flood\'", 2:"\'flood\'"}[class_of_interest]
    axes[0].set_title(f"Model prediction of {target_pred} patch\n(Explaining the {class_name} class in the red region)")
    features = [feature for feature in config["features"] if feature not in ["soil_class", "land_cover"]]
    feature_indices = {feature_name: list(range(index)) for feature_name, index in 
                    zip(["dem", "soil_bulk_density", "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "precipitation", 
                            "sentinel1", "flow_accumulation", "permanent_water", "flow_direction"], 
                            [1, 1, 2, 2, 12, 16, 3, 1, 1, 2])}
    s2_index, open_slice = None, 0
    for feature_name in features:
        number_channels = len(feature_indices[feature_name])
        if feature_name == "sentinel2":
            s2_index = open_slice
            break
        open_slice += number_channels
    if s2_index:
        s2_image = input_image[f"{target_map}_features"][0, (open_slice+2, open_slice+1, open_slice), :, :].permute(1, 2, 0).detach().cpu().numpy()
        s2_image = (s2_image - np.min(s2_image)) / (np.max(s2_image) - np.min(s2_image))
        axes[1].imshow(s2_image)
    grayscale_cam = np.ma.masked_where(grayscale_cam == 0, grayscale_cam)
    axes[1].imshow(grayscale_cam, alpha=0.4, cmap="jet", interpolation="bilinear")
    axes[1].set_title(f"Explanation by {method}\nfrom {target_map} feature map")
    fig.tight_layout()
    fig.savefig(f"{modelling_folder}/visualise/{config_name}.png", bbox_inches="tight")
    logger.info(f"Saved {method} explanation to: {modelling_folder}/visualise/{config_name}.png")
