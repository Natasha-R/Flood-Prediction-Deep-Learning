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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_data_loader(config, data_folder, ddp, subset=None, subevent=None, event=None, patch=None, mask_features=None, mask_patch=None):
     """
     Create a dataloader for a particular data subset, subevent or event.
     """
     dataset = FloodDataset(config, data_folder, subset, subevent, event, patch, mask_features, mask_patch)

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
     def __init__(self, config, data_folder, subset=None, subevent=None, event=None, patch=None, mask_features=None, mask_patch=None):
          
          self.config = config
          self.data_folder = data_folder
          self.scales = self.config["scales"]
          self.class_features = ["soil_class", "land_cover"]
          self.class_features_exist = config["class_features_exist"]
          self.indices_exist = "indices" in self.config["features"]
          self.features = [feature for feature in self.config["features"] if feature != "indices"]

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

          self.transform = transforms.Compose([ToTensor(config),
                                               Normalize(config, data_folder),
                                               MaskFeatures(config, mask_features),
                                               MaskPatch(config, mask_patch)])

     def __len__(self):
          return len(self.patches)
     
     def calculate_indices(self, index, scale):
          sentinel2 = tf.imread(f"{self.data_folder}/{scale}/sentinel2/{self.patches[index]}").astype(np.float32)
          ndvi = (sentinel2[:, :, 6] - sentinel2[:, :, 2]) / ((sentinel2[:, :, 6] + sentinel2[:, :, 2]) + np.finfo("float32").eps)
          ndmi = (sentinel2[:, :, 6] - sentinel2[:, :, 8]) / ((sentinel2[:, :, 6] + sentinel2[:, :, 8]) + np.finfo("float32").eps)
          ndwi = (sentinel2[:, :, 1] - sentinel2[:, :, 6]) / ((sentinel2[:, :, 1] + sentinel2[:, :, 6]) + np.finfo("float32").eps)
          cloud_time = sentinel2[:, :, 10:]
          return np.concatenate([np.expand_dims(ndvi, 2), np.expand_dims(ndmi, 2), np.expand_dims(ndwi, 2), cloud_time], axis=2)

     def get_data(self, index, scale, class_feature):

          data = [tf.imread(f"{self.data_folder}/{scale}/{feature}/{self.patches[index]}") for feature in self.features if ((feature in self.class_features) == class_feature)]
          if self.indices_exist:
               return data + [self.calculate_indices(index, scale)]
          return data
     
     def get_label(self, index, scale):
          return tf.imread(f"{self.data_folder}/{scale}/label/{self.patches[index]}")
     
     def __getitem__(self, index):

          data = {}

          for scale in self.scales:
               data[f"{scale}_features"] = self.get_data(index, scale, class_feature=False)
               data[f"{scale}_label"] = self.get_label(index, scale)
               if self.class_features_exist:
                    data[f"{scale}_classes"] = self.get_data(index, scale, class_feature=True)

          data = self.transform(data)

          return data

class ToTensor(object):

     def __init__(self, config):
          self.config = config
          self.scales = self.config["scales"]
          self.class_features_exist = config["class_features_exist"]

     def concat_data(self, features):
          return torch.concat([torch.from_numpy(np.expand_dims(feature, 0).astype(np.float32)) if feature.ndim == 2 else torch.from_numpy(feature.astype(np.float32)).permute(2, 0, 1) for feature in features], dim=0)
     
     def __call__(self, data):

          for scale in self.scales:
               
               data[f"{scale}_features"] = self.concat_data(data[f"{scale}_features"]) # CxHxW

               data[f"{scale}_label"] = torch.from_numpy(data[f"{scale}_label"].astype(np.int64))
               # originally: 0: no data, 1: aoi, 2: flood trace, 3: flooded area
               # after: 0: no data, 1: aoi, 2: flood
               if self.config.get("flood_trace_label", True):
                    data[f"{scale}_label"][data[f"{scale}_label"]==3] = 2 # combine flood trace and flooded
               else:
                    data[f"{scale}_label"][data[f"{scale}_label"]==2] = 1 # convert flood trace to other
                    data[f"{scale}_label"][data[f"{scale}_label"]==3] = 2
               
               if self.config.get("loss_function", "cross entropy").lower()=="dice":
                    data[f"{scale}_label"][data[f"{scale}_label"]==1] = 0
                    data[f"{scale}_label"][data[f"{scale}_label"]==2] = 1

               if self.class_features_exist:
                    data[f"{scale}_classes"] = self.concat_data(data[f"{scale}_classes"])

          return data

class Normalize(object):
     def __init__(self, config, data_folder):

          self.config = config
          self.scales = self.config["scales"]
          self.features = [feature for feature in config["features"] if feature not in ["soil_class", "land_cover"]]

          # import the metadata with the predefined shift and scale factors for each feature
          with open(f"{data_folder}/metadata/zscore.json") as file:
               self.zscore = json.load(file)

          # define the number of bands contained within each feature
          self.feature_indices = {feature_name: list(range(index)) for feature_name, index in 
                                  zip(["dem", "soil_bulk_density", "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "precipitation", 
                                       "sentinel1", "flow_accumulation", "permanent_water", "flow_direction", "soil_class", "land_cover",
                                       "indices", "summary_precipitation"], 
                                      [1, 1, 2, 2, 12, 16, 3, 1, 1, 2, 1, 1, 5, 3])}
          self.non_transformed_features = ["permanent_water", "soil_class", "land_cover", "indices"]

          # for each of the features selected for the model, create a tensor containing of the shift and scale factors for all of its bands
          self.zscore_values = {}
          self.features_to_transform = [feature for feature in self.features if not feature in self.non_transformed_features]
          for feature in self.features_to_transform:
               self.zscore_values[feature] = {"shift": torch.tensor([self.zscore[feature][str(band)]["shift"] for band in self.feature_indices[feature]]),
                                              "scale": torch.tensor([self.zscore[feature][str(band)]["scale"] for band in self.feature_indices[feature]])}
               
          # find the location (index range) within the tensor for all of the classes
          self.feature_channels = {}
          open_slice = 0
          
          for feature_name in self.features:
               number_channels = len(self.feature_indices[feature_name])
               self.feature_channels[feature_name] = slice(open_slice, open_slice + number_channels)
               open_slice += number_channels
               
     def apply_normalization(self, feature_name, feature):

          # these features do not need to be scaled
          if feature_name in self.non_transformed_features:
               return feature
          
          # apply log to features that need to be transformed
          if feature_name == "dem":
               feature = torch.log(torch.clamp(feature, min=-199, max=None) + 200)
          elif feature_name == "sentinel2":
               feature += 2
               for band in [0, 1, 2, 9]:
                    feature[band] = torch.log(feature[band])
          elif feature == "precipitation" or feature == "summary_precipitation":
               feature = torch.log(feature+1)
          
          return torch.clamp((feature - self.zscore_values[feature_name]["shift"][:, None, None]) / self.zscore_values[feature_name]["scale"][:, None, None], min=-3, max=3)
          
     def __call__(self, data):
          
          # normalize the features
          for feature_name in self.features:
               for scale in self.scales:
                    channels = self.feature_channels[feature_name]
                    data[f"{scale}_features"][channels] = self.apply_normalization(feature_name, data[f"{scale}_features"][channels])

          return data

class MaskFeatures(object):
     """
     Mask/permute (either shuffle or zero-out) the provided features.
     For the purpose of analysing the model behaviour with XAI.
     """
     def __init__(self, config, mask_features):

          self.config = config
          self.scales = self.config["scales"]
          self.features = [feature for feature in config["features"] if feature not in ["soil_class", "land_cover"]]
          self.class_features = [feature for feature in config["features"] if feature in ["soil_class", "land_cover"]]
          self.feature_indices = {feature_name: list(range(index)) for feature_name, index in 
                                  zip(["dem", "soil_bulk_density", "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "precipitation", 
                                       "sentinel1", "flow_accumulation", "permanent_water", "flow_direction", "soil_class", "land_cover",
                                       "indices", "summary_precipitation"], 
                                       [1, 1, 2, 2, 12, 16, 3, 1, 1, 2, 1, 1, 5, 3])}
          self.mask_features = mask_features

          self.feature_channels, self.class_feature_channels = {}, {}
          for feature_group, feature_channels in zip([self.features, self.class_features], [self.feature_channels, self.class_feature_channels]):
               open_slice = 0
               for feature_name in feature_group:
                    number_channels = len(self.feature_indices[feature_name])
                    feature_channels[feature_name] = slice(open_slice, open_slice + number_channels)
                    open_slice += number_channels

     def __call__(self, data):

          if self.mask_features:
               for scale in self.mask_features:
                    for feature_name in self.mask_features[scale]:

                         if feature_name in self.feature_channels:
                              feature_channels = self.feature_channels
                              key = "features"
                         else: # if feature name is a class feature
                              feature_channels = self.class_feature_channels
                              key = "classes"

                         channels = feature_channels[feature_name]

                         for channel in range(data[f"{scale}_{key}"][channels].shape[0]):
                              if not self.config["permute"]:
                                   data[f"{scale}_{key}"][channels][channel, :, :] = 0
                              else:
                                   data[f"{scale}_{key}"][channels][channel, :, :] = data[f"{scale}_{key}"][channels][channel, :, :].view(-1)[torch.randperm(256*256)].view(1, 256, 256) 
               return data
          
          else:
               return data
          
class MaskPatch(object):
     """
     Mask/permute (either shuffle or zero-out) the provided patch coordinates.
     For the purpose of analysing the model behaviour with XAI.
     """
     def __init__(self, config, mask_patch):

          self.config = config
          self.mask_patch = mask_patch
          
     def __call__(self, data):

          if self.mask_patch:

               scale = self.mask_patch["scale"]
               top_left_x, top_left_y = [int(value) for value in self.mask_patch["top_left"]]
               bottom_right_x, bottom_right_y = [int(value) for value in self.mask_patch["bottom_right"]]

               for key in ["features", "classes"]:
                    if not self.config["permute"]:
                         data[f"{scale}_{key}"][:, top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
                    else:
                         for channel in range(data[f"{scale}_{key}"].shape[0]):
                              data[f"{scale}_{key}"][channel, top_left_y:bottom_right_y, top_left_x:bottom_right_x] = \
                              data[f"{scale}_{key}"][channel, top_left_y:bottom_right_y, top_left_x:bottom_right_x] \
                              .contiguous().view(-1)[torch.randperm((bottom_right_x-top_left_x)*(bottom_right_y-top_left_y))] \
                              .view(1, bottom_right_y-top_left_y, bottom_right_x-top_left_x) 

               return data
     
          else:
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
                 zip(["dem", "soil_bulk_density", "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "sentinel1", "flow_accumulation", "summary_precipitation"], 
                     [1, 1, 2, 2, 12, 3, 1, 3]) 
                 for band_index in range(feature_count)]
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
                    raster = np.log(raster + 200)
               elif feature == "sentinel2":
                    raster = raster + 2
                    if band == 0 or band == 1 or band == 2 or band == 9:
                         raster = np.log(raster)
               elif feature == "precipitation" or feature == "summary_precipitation":
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

     # permanently save the values in a json file
     with open(f"{data_folder}/metadata/zscore.json", 'w') as file:
          json.dump(zscore, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--normalize", action="store_true", default=False, help="Calculate the normalization of the dataset.")
    args = parser.parse_args()

    if args.normalize:
        normalize(data_folder=args.data_folder)