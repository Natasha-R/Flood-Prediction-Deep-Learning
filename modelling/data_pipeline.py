import torch
from torchvision import transforms
import pandas as pd
import tifffile as tf
import numpy as np

class FloodDataset(torch.utils.data.Dataset):
     """
     The dataset for the flood data.
     """
     def __init__(self, config, data_folder, subset=None, subevent=None, event=None):
          
          self.config = config
          self.data_folder = data_folder

          # subset the data patches based on a particular train/validation split, subevent, or event
          data_subset = pd.read_csv(f"{data_folder}/metadata/data_subset.csv")
          if subset:
                data_subset = data_subset[data_subset["subset"]==subset]
          if subevent:
               data_subset = data_subset[data_subset["subevent"]==subevent]
          if event: 
               data_subset = data_subset[data_subset["event"]==event]
          self.patches = list(data_subset["patch"])

          self.transform = transforms.Compose([ToTensor(),
                                               ProcessLabel(config),
                                               Normalize(config)])

     def __len__(self):
          return len(self.patches)
     
     def __getitem__(self, index):

          local_label = tf.imread(f"{self.data_folder}/local/label/{self.patches[index]}") # HxW
          local_features = {feature : tf.imread(f"{self.data_folder}/local/{feature}/{self.patches[index]}") for feature in self.config["features"]} # HxWxC

          data = {"local_features": local_features, 
                  "local_label": local_label}

          data = self.transform(data)

          return data

def create_data_loader(config, data_folder, subset=None, subevent=None, event=None):
     """
     Create a dataloader for a particular data subset, subevent or event.
     """
     dataset = FloodDataset(config=config, data_folder=data_folder, subset=subset, subevent=subevent, event=event)
     loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config["batch_size"],
                                             num_workers=config["number_workers"],
                                             shuffle=True,
                                             pin_memory=True)
     return loader

def create_data_loaders(config, data_folder):
     """
     Create train and validation dataloaders for the flood dataset.
     """
     train_dataset = FloodDataset(config=config, data_folder=data_folder, subset="train")
     validation_dataset = FloodDataset(config=config, data_folder=data_folder, subset="validation")
     
     train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config["batch_size"],
                                                num_workers=config["number_workers"],
                                                shuffle=True,
                                                pin_memory=True)
     
     validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                     batch_size=config["batch_size"],
                                                     num_workers=config["number_workers"],
                                                     shuffle=True,
                                                     pin_memory=True)

     return train_loader, validation_loader

class ToTensor(object):
     def __call__(self, data):
          data["local_features"] = {feature_name : torch.from_numpy(np.expand_dims(feature, 0).astype(np.float32)) 
                                    if feature.ndim == 2 
                                    else torch.from_numpy(feature.astype(np.float32)).permute(2, 0, 1) 
                                    for feature_name, feature in data["local_features"].items()} # CxHxW
          data["local_label"] = torch.from_numpy(data["local_label"].astype(np.int64)) # HxW
          return data

class ProcessLabel(object):
     def __init__(self, config):
          # 0: no data, 1: aoi, 2: flood trace, 3: flooded area
          self.config = config
     def __call__(self, data):
          if not self.config["separate_flood_trace_label"]:
               data["local_label"][data["local_label"]==3] = 2
          return data

class Normalize(object):
     def __init__(self, config):
          None
     def __call__(self, data):
          return data