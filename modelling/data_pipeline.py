import torch
from torchvision import transforms
import pandas as pd
import tifffile as tf
import numpy as np
import json
import os
from collections import defaultdict

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
                                               Normalize(config, data_folder)])

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
     def __init__(self, config, data_folder):

          # import the metadata with the predefined shift and scale factors for each feature
          with open(f"{data_folder}/metadata/zscore.json") as file:
               self.zscore = json.load(file)

          # define the number of bands contained within each feature
          self.feature_indices = {feature_name: list(range(index)) for feature_name, index in 
                                  zip(["dem", "soil_bulk_density", "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "precipitation"], 
                                      [1, 1, 2, 2, 10, 16])}
          
          # for each of the features selected for the model, create a tensor containing of the shift and scale factors for all of its bands
          def nested_defaultdict():
               return defaultdict(nested_defaultdict)
          self.zscore_values = nested_defaultdict()
          features_to_transform = [feature for feature in config["features"] if not feature in ["permanent_water", "soil_class"]]
          for feature in features_to_transform:
               self.zscore_values[feature]["shift"] = torch.tensor([self.zscore[feature][str(band)]["shift"] for band in self.feature_indices[feature]])
               self.zscore_values[feature]["scale"] = torch.tensor([self.zscore[feature][str(band)]["scale"] for band in self.feature_indices[feature]])

     def apply_normalization(self, feature_name, feature):

          # these features do not need to be scaled
          if feature_name in ["permanent_water", "soil_class"]:
               return feature
          
          # apply log to features that need to be transformed
          if feature_name == "dem":
               feature = torch.log(feature + 200)
          elif feature_name == "sentinel2":
               feature += 2
               for band in [0, 1, 2, 9]:
                    feature[band] = torch.log(feature[band])
          elif feature == "precipitation":
               feature = torch.log(feature+1)

          # scale the data using the predefined shift and scale factors, performing either z-normalisation or min-max normalization
          # clip the data to -3/+3 (for the z-normalized features)
          return torch.clamp((feature - self.zscore_values[feature_name]["shift"][:, None, None]) / self.zscore_values[feature_name]["scale"][:, None, None], min=-3, max=3)
          
     def __call__(self, data):
          
          # normalize the features
          data["local_features"] = {feature_name : self.apply_normalization(feature_name, feature) for feature_name, feature in data["local_features"].items()}

          return data
     
def normalize(data_folder=os.environ["DATA_FOLDER"]):
     """
     A function to be run before model training, to calculate the shift and scale values for normalizing the data.
     """
     def nested_defaultdict():
          return defaultdict(nested_defaultdict)
     zscore = nested_defaultdict()

     # define the features to normalize
     features = [(feature_name, band_index) for feature_name, feature_count in 
                 zip(["dem", "soil_bulk_density", "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2"], 
                     [1, 1, 2, 2, 12]) 
                 for band_index in range(feature_count)]
     features = features + [("precipitation", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)), ("precipitation", 14), ("precipitation", 15)]

     for feature, band in features:

          paths = [path.path for path in os.scandir(f"{data_folder}/full_subevent/raster_{feature}/") if path.path.endswith(".tif")]
          summed, summed_squared, total_values = 0, 0, 0
          minimum, maximum = None, None

          # read in each subevent
          for path in paths:
               raster = tf.imread(path)
               reference = tf.imread(f"{data_folder}/full_subevent/raster_cems/{path.split('/')[-1]}")
               if raster.ndim == 2: raster = np.expand_dims(raster, axis=-1)
               try:
                    raster = raster[:, :, band]
               except:
                    print(f"Issue with {path} for {feature} (band {band})", flush=True)
                    continue
               if raster.shape[:2] != reference.shape:
                    print(f"Issue with {path} for {feature} (band {band})", flush=True)
                    continue
               # utilise only data within the AOIs
               raster = raster[reference != 0]

               # apply log to features that need to be transformed
               if feature == "dem":
                    raster = np.log(raster + 200)
               elif feature == "sentinel2":
                    raster = raster + 2
                    if band == 0 or band == 1 or band == 2 or band == 9:
                         raster = np.log(raster)
               elif feature == "precipitation":
                    raster = np.log(raster+1)
                    
               # save the sum of the values, sum of their squares, and total number of values
               raster = raster.astype(np.float64)
               total_values += len(raster.flatten())
               summed += np.sum(raster)
               summed_squared += np.sum(raster**2)
               if not minimum or np.min(raster) < minimum:
                    minimum = np.min(raster)
               if not maximum or np.max(raster) > maximum:
                    maximum = np.max(raster)

          # save the shift (mean) and scale (standard deviation) for each of the feature bands
          if not (feature == "precipitation" and isinstance(band, tuple)):
               band = (band,)
          for band_index in band:
               zscore[feature][band_index]["shift"] = summed.item()/total_values
               zscore[feature][band_index]["scale"] = np.sqrt((summed_squared.item()/total_values) - ((summed.item()/total_values) ** 2))
               zscore[feature][band_index]["min"] = minimum
               zscore[feature][band_index]["max"] = maximum

     # for the feature bands that use mix/max scaling instead of z-normalisation, set their shift to 0 and scale to the maximum
     for band in range(16):
          zscore["precipitation"][band]["shift"] = 0
          zscore["precipitation"][band]["scale"] = zscore["precipitation"][band]["max"]
     for band in [10, 11]:
          zscore["sentinel2"][band]["shift"] = 1
          zscore["sentinel2"][band]["scale"] = zscore["sentinel2"][band]["max"]

     # permanently save the values in a json file
     with open(f"{data_folder}/metadata/zscore.json", 'w') as file:
          json.dump(zscore, file)
          