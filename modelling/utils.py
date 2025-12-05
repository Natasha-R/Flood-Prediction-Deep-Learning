import logging
import torch
import os
import yaml
import torch.nn as nn
import sys
import modelling.architectures as architectures

def get_logger():

    logger = logging.getLogger("log")
    logger.propagate = False
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=f"logger.log")
    file_handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

def check_paths(args):
    # check if the config file, data folder and modelling folder paths exist
    for path in [args.config_path, args.data_folder, args.modelling_folder]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path: {path} does not exist")
        
def check_cuda():
    # check if cuda is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")
    
def load_config(config_path, logger=None):
    """
    Load in and process the modelling configuration file.
    """
    if logger:
        logger.info(f"Reading from config: {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config_name = config_path.split("/")[-1].split(".")[0]
    config["num_classes"] = 4 if config["separate_flood_trace_label"] else 3
    class_features = ["soil_class", "land_cover"]
    config["class_features_exist"] = any(feature in class_features for feature in config["features"])

    return config, config_name

def load_model(config, device, logger, pretrained_path=None):

    # load the model architecture
    if config["architecture"].lower()=="basicunet":
        model = architectures.BasicUNet(config)
    elif config["architecture"].lower()=="chainedunet":
        model = architectures.ChainedUNet(config)
    else:
        raise ValueError(f"Unrecognised model name: '{config['architecture']}'")

    # load in any pretrained model weights
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, weights_only=False, map_location=torch.device(device)))

    # put the model onto the GPU(s)
    device_ids = list(range(torch.cuda.device_count()))
    device_names = [torch.cuda.get_device_name(device_id) for device_id in device_ids]
    if device == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        device_ids = [int(device[-1])]
        device_names = device_names[int(device[-1])]
    model = model.to(device)

    logger.info(f"Model loaded onto: {device_names}")

    return model

def find_num_channels(config):
    channels_in_features = {"dem":1, "permanent_water":1, "soil_bulk_density":1, "flow_accumulation":1,
                                "soil_moisture_one_week":2, "soil_moisture_one_day":2, "soil_class":3,
                                "precipitation":16, "sentinel1":3, "sentinel2":12, "flow_direction":2}
    in_channels = sum([channels_in_features[feature] for feature in config["features"]])
    in_embeddings = sum([channels_in_features[feature] for feature in config["features"] if "class" in feature])
    return in_channels, in_embeddings