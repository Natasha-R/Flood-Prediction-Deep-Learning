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
    
def load_config(config_path, logger):
    """
    Load in and process the modelling configuration file.
    """
    logger.info(f"Reading from config: {config_path}")
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config_name = config_path.split("/")[-1].split(".")[0]
    config["num_classes"] = 4 if config["separate_flood_trace_label"] else 3

    return config, config_name

def load_model(config, device, logger, pretrained_path=None):

    # load the model architecture
    if config["architecture"].lower()=="test":
        model = architectures.TestModel(config, device)
    elif config["architecture"].lower()=="basicunet":
        model = architectures.BasicUNet(config, device)

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