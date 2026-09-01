import torch
from torchvision import transforms
import pandas as pd
import tifffile as tf
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import rasterio
from modelling import utils

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_data_loader(config, data_folder, ddp, subset=None, subevent=None, event=None, patch=None, mask_features=None, mask_feature_bands=False, training=False, mask_before_normalization=False, mask_value=0):
     """
     Create a dataloader for a particular data subset, subevent or event.
     """
     dataset = FloodDataset(config, data_folder, subset, subevent, event, patch, mask_features, mask_feature_bands, training, mask_before_normalization, mask_value)

     if ddp: 
          loader = torch.utils.data.DataLoader(
          dataset=dataset,
          batch_size=config["batch_size"],
          num_workers=config["number_workers"],
          shuffle=False,
          sampler=DistributedSampler(dataset=dataset, shuffle=True))
     else:
          generator = torch.Generator()
          generator.manual_seed(47)
          loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config["batch_size"],
                                             num_workers=config["number_workers"],
                                             shuffle=True,
                                             pin_memory=True,
                                             worker_init_fn=seed_worker,
                                             generator=generator,)

     return loader

class FloodDataset(torch.utils.data.Dataset):
     """
     The dataset for the flood data.
     """
     def __init__(self, config, data_folder, subset=None, subevent=None, event=None, patch=None, mask_features=None, mask_feature_bands=False, training=False, mask_before_normalization=False, mask_value=0):
          
          self.config = config
          self.data_folder = data_folder
          self.class_features = utils.get_class_features()
          self.derived_features = utils.get_derived_features()

          self.scales = self.config["scales"]
          self.use_consistent_context = self.config.get("use_consistent_context", False)
          self.features = {scale: [feature for feature in self.config[f"{scale}_features"]] for scale in self.scales}
          self.predict_feature = self.config.get("predict_feature", False)
          self.class_exists = {f"{scale}": any([feature for feature in self.config[f"{scale}_features"] if feature in self.class_features]) for scale in self.scales}
          self.pixel_locations = np.arange(256, dtype=np.float32) + 0.5

          # subset the data patches based on a particular train/validation split, subevent, or event
          data_subset = pd.read_csv(f"{data_folder}/subsets/{config['data_subset_file']}.csv")
          if subset:
               data_subset = data_subset[data_subset["subset"]==subset]
          if subevent:
               data_subset = data_subset[data_subset["subevent"]==subevent]
          if event: 
               data_subset = data_subset[data_subset["event"]==event]
          if patch:
               data_subset = data_subset[data_subset["patch"]==patch]
          self.data_subset = data_subset
          self.patches = list(data_subset["patch"])
          self.subevents = list(data_subset["subevent"])

          transform_operations = [ToTensor(config), 
                                  Normalize(config, data_folder),
                                  HorizontalFlip(training, subset)]
          transform_operations.insert(1 if mask_before_normalization else 2, MaskFeatures(config, mask_features, mask_feature_bands, mask_value))
          self.transform = transforms.Compose(transform_operations)

     def __len__(self):
          return len(self.patches)
     
     def calculate_indices(self, index, scale):
          sentinel2 = tf.imread(f"{self.data_folder}/{scale}/sentinel2/{self.patches[index]}").astype(np.float32)
          ndvi = (sentinel2[:, :, 6] - sentinel2[:, :, 2]) / ((sentinel2[:, :, 6] + sentinel2[:, :, 2]) + np.finfo("float32").eps)
          ndmi = (sentinel2[:, :, 6] - sentinel2[:, :, 8]) / ((sentinel2[:, :, 6] + sentinel2[:, :, 8]) + np.finfo("float32").eps)
          ndwi = (sentinel2[:, :, 1] - sentinel2[:, :, 6]) / ((sentinel2[:, :, 1] + sentinel2[:, :, 6]) + np.finfo("float32").eps)
          cloud_time = sentinel2[:, :, 10:]
          return np.concatenate([np.expand_dims(ndvi, 2), np.expand_dims(ndmi, 2), np.expand_dims(ndwi, 2), cloud_time], axis=2)

     def calculate_coords(self, index, scale):
          with rasterio.open(f"{self.data_folder}/{scale}/label/{self.patches[index]}") as src:
               transform = src.transform
          return np.stack([np.broadcast_to(((transform.c + self.pixel_locations * transform.a) / 180.0)[None, :], (256, 256)), 
                           np.broadcast_to(((transform.f + self.pixel_locations * transform.e) / 90.0)[:, None], (256, 256))], axis=-1)

     def get_data(self, index, scale, class_feature):

          scale_folder = "con_context" if scale == "context" and self.use_consistent_context else scale
          # imported features
          data = [tf.imread(f"{self.data_folder}/{scale_folder}/{feature}/{self.patches[index]}") for feature in self.features[scale] if\
                  (((feature in self.class_features) == class_feature) and (feature not in self.derived_features))]
          # derived features
          if "indices" in self.features[scale]:
               data = data + [self.calculate_indices(index, scale_folder)]
          if "coordinates" in self.features[scale]:
               data = data + [self.calculate_coords(index, scale_folder)]
          return data
     
     def get_label(self, index, scale):
          scale_folder = "con_context" if (scale == "context") and self.use_consistent_context else scale
          return tf.imread(f"{self.data_folder}/{scale_folder}/label/{self.patches[index]}")
     
     def get_img_label(self, index, feature):
          return tf.imread(f"{self.data_folder}/local/{feature}/{self.patches[index]}")

     def __getitem__(self, index):

          data = {}

          for scale in self.scales:
               data[f"{scale}_features"] = self.get_data(index, scale, class_feature=False)
               data[f"{scale}_label"] = self.get_label(index, scale) if not self.predict_feature else self.get_img_label(index, self.predict_feature)
               if self.class_exists[scale]:
                    data[f"{scale}_classes"] = self.get_data(index, scale, class_feature=True)
          data["metadata"] = self.subevents[index]

          data = self.transform(data)

          return data

class ToTensor(object):

     def __init__(self, config):
          self.config = config
          self.scales = self.config["scales"]
          self.class_features = utils.get_class_features()
          self.class_exists = {f"{scale}": any([feature for feature in self.config[f"{scale}_features"] if feature in self.class_features]) for scale in self.scales}
          self.predict_feature = self.config.get("predict_feature", False)
          
     def concat_data(self, features):
          return torch.concat([torch.from_numpy(np.expand_dims(feature, 0).astype(np.float32)) if feature.ndim == 2 else torch.from_numpy(feature.astype(np.float32)).permute(2, 0, 1) for feature in features], dim=0)
     
     def __call__(self, data):

          for scale in self.scales:
               
               # features
               data[f"{scale}_features"] = self.concat_data(data[f"{scale}_features"]) # CxHxW
               if self.class_exists[scale]:
                    data[f"{scale}_classes"] = self.concat_data(data[f"{scale}_classes"])

               # label
               if not self.predict_feature:
                    data[f"{scale}_label"] = torch.from_numpy(data[f"{scale}_label"].astype(np.int64))
                    # originally: 0: no data, 1: aoi, 2: flood trace, 3: flooded area # after: 0: no data, 1: aoi, 2: flood
                    if self.config.get("flood_trace_label", True):
                         data[f"{scale}_label"][data[f"{scale}_label"]==3] = 2 # convert flood trace and flooded to "flood"
                    else:
                         data[f"{scale}_label"][data[f"{scale}_label"]==2] = 1 # convert flood trace to "aoi" (non-flood)
                         data[f"{scale}_label"][data[f"{scale}_label"]==3] = 2
                    # if using dice loss, convert the label to binary
                    if self.config.get("loss_function", "cross entropy").lower()=="dice":
                         data[f"{scale}_label"][data[f"{scale}_label"]==1] = 0
                         data[f"{scale}_label"][data[f"{scale}_label"]==2] = 1
               else:
                    data[f"{scale}_label"] = torch.from_numpy(data[f"{scale}_label"].astype(np.float32))

               # convert label to a classification value
               if self.config.get("classification_threshold", False):
                    proportion_flood = (data[f"{scale}_label"] == 2).sum() / ((data[f"{scale}_label"] != 0).sum())
                    if proportion_flood > self.config["classification_threshold"]:
                         data[f"{scale}_label"] = torch.tensor(2, dtype=torch.int64)
                    else:
                         data[f"{scale}_label"] = torch.tensor(1, dtype=torch.int64)

          return data
     
class HorizontalFlip(object):

     def __init__(self, training, subset):
          self.training = training
          self.subset = subset

     def __call__(self, data):
          # randomly flip the patches during training
          if self.training and self.subset=="train" and random.choice([True, False]):
               for key in data:
                    if "metadata" not in key:
                         data[key] = torch.flip(data[key], dims=[-1])
          return data

class Normalize(object):
     def __init__(self, config, data_folder):

          self.config = config
          self.scales = self.config["scales"]
          self.predict_feature = self.config.get("predict_feature", False)

          # import the metadata with the predefined shift and scale factors for each feature
          self.relative_dem = self.config.get("use_relative_dem", False)
          if self.relative_dem:
               with open(f"{data_folder}/metadata/dem_subevents.json") as file:
                    self.dem_subevents = json.load(file)
               with open(f"{data_folder}/metadata/zscore_with_dem.json") as file:
                    self.zscore = json.load(file)
          else:
               with open(f"{data_folder}/metadata/zscore.json") as file:
                    self.zscore = json.load(file)

          self.class_features = utils.get_class_features()
          self.non_transformed_features = utils.get_non_transformed_features()

          self.non_class_features = {scale: [feature for feature in self.config[f"{scale}_features"] if feature not in self.class_features] for scale in self.scales}
          self.features_to_transform = {scale: [feature for feature in self.non_class_features[scale] if feature not in self.non_transformed_features] for scale in self.scales}

          # for each of the features selected for the model, create a tensor containing of the shift and scale factors for all of its bands
          self.zscore_values = {}
          self.feature_channels = {}
          self.feature_indices = utils.get_indices_per_feature()
          for scale in self.scales:
               self.zscore_values[scale] = {}
               for feature in self.features_to_transform[scale]:
                    self.zscore_values[scale][feature] = {"shift": torch.tensor([self.zscore[feature][str(band)]["shift"] for band in self.feature_indices[feature]]),
                                                          "scale": torch.tensor([self.zscore[feature][str(band)]["scale"] for band in self.feature_indices[feature]])}
               # find the location (index range) within the tensor for all of the classes
               open_slice = 0
               self.feature_channels[scale] = {}
               for feature_name in self.non_class_features[scale]:
                    number_channels = len(self.feature_indices[feature_name])
                    self.feature_channels[scale][feature_name] = slice(open_slice, open_slice + number_channels)
                    open_slice += number_channels

     def apply_normalization(self, scale, feature_name, feature, subevent):

          # these features do not need to be scaled
          if feature_name in self.non_transformed_features:
               return feature
          
          # apply log to features that need to be transformed
          if feature_name == "dem":
               if self.relative_dem:
                    if subevent in self.dem_subevents:
                         feature += self.dem_subevents[subevent]
               feature = torch.log(torch.clamp(feature, min=-199, max=None) + 200)
          elif feature_name == "sentinel2":
               feature += 2
               for band in [0, 1, 2, 9]:
                    feature[band] = torch.log(feature[band])
          elif feature_name in ["precipitation", "summary_precipitation", "slope", "hand"]:
               feature = torch.log(feature+1)

          return torch.clamp((feature - self.zscore_values[scale][feature_name]["shift"][:, None, None]) / self.zscore_values[scale][feature_name]["scale"][:, None, None], min=-3, max=3)
          
     def __call__(self, data):
          
          # normalize the features
          for scale in self.scales:
               for feature_name in self.non_class_features[scale]:
                    channels = self.feature_channels[scale][feature_name]
                    data[f"{scale}_features"][channels] = self.apply_normalization(scale, feature_name, data[f"{scale}_features"][channels], data["metadata"])

               if self.predict_feature:
                    data[f"{scale}_label"] = self.apply_normalization(scale, self.predict_feature, data[f"{scale}_label"], data["metadata"])

          return data

class MaskFeatures(object):
     """
     Mask/permute (either shuffle or zero-out) the provided features.
     For the purpose of analysing the model behaviour with XAI.
     """
     def __init__(self, config, mask_features, mask_feature_bands, mask_value):

          self.config = config
          self.scales = self.config["scales"]
          self.class_features = utils.get_class_features()

          self.non_class_features = {scale: [feature for feature in self.config[f"{scale}_features"] if feature not in self.class_features] for scale in self.scales}
          self.class_features = {scale: [feature for feature in self.config[f"{scale}_features"] if feature in self.class_features] for scale in self.scales}
          self.feature_indices = utils.get_indices_per_feature()
          self.mask_features = mask_features
          self.mask_feature_bands = mask_feature_bands
          self.mask_value = mask_value

          self.feature_channels, self.class_feature_channels = {}, {}
          for scale in self.scales:
               self.feature_channels[scale] = {}
               self.class_feature_channels[scale] = {}
               for feature_group, feature_channels in zip([self.non_class_features[scale], self.class_features[scale]], [self.feature_channels[scale], self.class_feature_channels[scale]]):
                    open_slice = 0
                    for feature_name in feature_group:
                         number_channels = len(self.feature_indices[feature_name])
                         feature_channels[feature_name] = slice(open_slice, open_slice + number_channels)
                         open_slice += number_channels

     def __call__(self, data):

          if self.mask_features:
               for scale in self.mask_features:
                    for feature_name in self.mask_features[scale]:

                         if self.mask_feature_bands: 
                              feature_band = feature_name[1]
                              feature_name = feature_name[0]                                   

                         if feature_name in self.feature_channels[scale]:
                              feature_channels = self.feature_channels[scale]
                              key = "features"
                         else: # if feature name is a class feature
                              feature_channels = self.class_feature_channels[scale]
                              key = "classes"

                         channels = feature_channels[feature_name]

                         for channel in range(data[f"{scale}_{key}"][channels].shape[0]):
                              if self.mask_feature_bands:
                                   channel = feature_band
                              if self.mask_value == "permute":
                                   data[f"{scale}_{key}"][channels][channel, :, :] = data[f"{scale}_{key}"][channels][channel, :, :].view(-1)[torch.randperm(256*256)].view(1, 256, 256) 
                              else:
                                   data[f"{scale}_{key}"][channels][channel, :, :] = float(self.mask_value)
                              if self.mask_feature_bands:
                                   break

          return data

def normalize(data_folder=os.environ["DATA_FOLDER"]):
     """
     A function to be run before model training, to calculate the shift and scale values for normalizing the data.
     """
     def nested_defaultdict():
          return defaultdict(nested_defaultdict)
     zscore = nested_defaultdict()

     # define the features to normalize
     features = [(feature_name, index) for feature_name, indices in utils.get_indices_per_feature().items() for index in indices if feature_name != "precipitation"]
     features = features + [("precipitation", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)), ("precipitation", 14), ("precipitation", 15)]

     for feature, band in tqdm(features, desc="Feature"):

          paths = [path.path for path in os.scandir(f"{data_folder}/full_subevent/raster_{feature}/") if path.path.endswith(".tif")]
          summed, summed_squared, total_values = 0, 0, 0
          minimum, maximum = None, None

          # read in each subevent
          for path in tqdm(paths, desc="Path"):
               raster = tf.imread(path)
               reference = tf.imread(f"{data_folder}/full_subevent/raster_cems/{path.split('/')[-1]}")
               if raster.ndim == 2: raster = np.expand_dims(raster, axis=-1)
               raster = raster[:, :, band]
               
               # utilise only data within the AOIs
               raster = raster[reference != 0]

               # apply log to features that need to be transformed
               if feature == "dem":
                    with open(f"{data_folder}/metadata/dem_subevents.json") as file:
                         dem_subevents = json.load(file)
                    if path.split("/")[-1][:-4] in dem_subevents:
                         raster += dem_subevents[path.split("/")[-1][:-4]]
                    raster = torch.log(raster + 200)
               elif feature == "sentinel2":
                    raster = raster + 2
                    if band == 0 or band == 1 or band == 2 or band == 9:
                         raster = np.log(raster)
               elif feature in ["precipitation", "summary_precipitation", "slope", "hand"]:
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
     for band in range(3):
          zscore["summary_precipitation"][band]["shift"] = 0
          zscore["summary_precipitation"][band]["scale"] = zscore["summary_precipitation"][band]["max"]
     for band in [10, 11]:
          zscore["sentinel2"][band]["shift"] = 1
          zscore["sentinel2"][band]["scale"] = zscore["sentinel2"][band]["max"]
     zscore["sentinel1"][2]["shift"] = 0
     zscore["sentinel1"][2]["scale"] = zscore["sentinel1"][2]["max"]
     for band in [0, 1]:
          zscore["flow_direction"][band]["shift"] = 0
          zscore["flow_direction"][band]["scale"] = 10000
     zscore["slope"][0]["shift"] = 0
     zscore["slope"][0]["scale"] = zscore["slope"][0]["max"]
     zscore["hand"][0]["shift"] = 0
     zscore["hand"][0]["scale"] = zscore["hand"][0]["max"]

     # permanently save the values in a json file
     with open(f"{data_folder}/metadata/zscore_new.json", 'w') as file:
          json.dump(zscore, file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--normalize", action="store_true", default=False, help="Calculate the normalization of the dataset.")
    args = parser.parse_args()

    if args.normalize:
        normalize(data_folder=args.data_folder)