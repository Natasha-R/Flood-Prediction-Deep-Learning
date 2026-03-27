from tqdm import tqdm
import os
import pandas as pd
from functools import reduce
import numpy as np
import tifffile as tf
import argparse
from scipy import stats
from collections import defaultdict
import geopandas as gpd
from itertools import combinations
from sklearn.metrics import mutual_info_score
from dcor import distance_correlation

def calculate_dataset_distributions(data_folder, scale):
    """
    Calculate the distributions of each of the features.
    """

    features = [(feature_name, band_index) for feature_name, feature_count in zip(["dem", "permanent_water", "soil_bulk_density", "soil_class", "label", "flow_accumulation",
                                                                                   "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "sentinel1"],
                                                                                   [1, 1, 1, 1, 1, 1, 2, 2, 12, 3]) for band_index in range(feature_count)]
    features = features + [("precipitation", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)), ("precipitation", 14), ("precipitation", 15)]
    for feature, band in tqdm(features):
        if scale == "local":
            paths = [path.path for path in os.scandir(f"{data_folder}/full_subevent/raster_{feature}/") if path.path.endswith(".tif")]
        else:
            paths = [path.path for path in os.scandir(f"{data_folder}/{scale}/{feature}/") if path.path.endswith(".tif")]
        feature_value_counts = []

        for path in paths:
            raster = tf.imread(path)
            if raster.ndim == 2: raster = np.expand_dims(raster, axis=-1)
            raster = raster[:, :, band]
            if scale == "local":
                reference = tf.imread(f"{data_folder}/full_subevent/raster_cems/{path.split('/')[-1]}")
                raster = raster[reference != 0]
            feature_value_counts.append(pd.Series(raster.flatten()).value_counts())

        feature_value_counts = pd.DataFrame(reduce(lambda a, b: a.add(b, fill_value=0), feature_value_counts).astype(int)).reset_index(names="value")
        feature_value_counts["feature"] = f"{feature}_{band}"
        feature_value_counts_path = f"{data_folder}/metadata/dataset_distributions_{scale}.csv"
        feature_value_counts.to_csv(feature_value_counts_path, mode="a", header=not os.path.exists(feature_value_counts_path), index=False)

def calculate_dataset_associations(data_folder):
    """
    Calculate the correlations, mutual information and distance correlations between all of the features.
    """

    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    features = [(feature_name, band_index) for feature_name, feature_count in zip(["dem", "permanent_water", "soil_bulk_density", "flow_accumulation", "soil_moisture_one_day", 
                                                                                   "soil_moisture_one_week", "sentinel2", "sentinel1", "flow_direction", "precipitation"],
                                                                                   [1, 1, 1, 1, 2, 2, 11, 2, 2, 16]) for band_index in range(feature_count)]
    associations = {f"{class_type}_{algorithm}": defaultdict(list) for class_type in ["all", "flood", "non_flood"] 
                    for algorithm in ["correlation", "mutual_information", "distance_correlation"]}

    for index in tqdm(range(len(raster_extents))):

        subevent = raster_extents.loc[index, "subevent"]
        feature_data = {"all":{}, "flood":{}, "non_flood":{}}

        for feature_name, band_index in features:

            data = tf.imread(f"{data_folder}/full_subevent/raster_{feature_name}/{subevent}.tif")
            reference = tf.imread(f"{data_folder}/full_subevent/raster_label/{subevent}.tif")

            if data.ndim == 2: data = np.expand_dims(data, axis=-1)
            data = data[:, :, band_index]
            all_data = data[reference != 0]
            flood_data = data[(reference == 2) | (reference == 3)]
            non_flood_data = data[reference == 1]

            feature_data["all"][f"{feature_name}_{band_index}"] = all_data
            feature_data["flood"][f"{feature_name}_{band_index}"] = flood_data
            feature_data["non_flood"][f"{feature_name}_{band_index}"] = non_flood_data

        for class_type in ["all", "flood", "non_flood"]:
            for feature_a, feature_b in combinations(feature_data[class_type].keys(), 2):
                associations[f"{class_type}_correlation"][f"{feature_a}-{feature_b}"].append(stats.pearsonr(feature_data[class_type][feature_a], feature_data[class_type][feature_b]).statistic.item())
                associations[f"{class_type}_mutual_information"][f"{feature_a}-{feature_b}"].append(mutual_info_score(feature_data[class_type][feature_a], feature_data[class_type][feature_b]))
                associations[f"{class_type}_distance_correlation"][f"{feature_a}-{feature_b}"].append(distance_correlation(feature_data[class_type][feature_a].astype(float), feature_data[class_type][feature_b].astype(float)).item())

        for key in associations:
            pd.DataFrame(associations[key]).to_csv(f"{data_folder}/metadata/associations/feature_{key}.csv", index=False)

    print("Complete!", flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the distributions of each of the features.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument('--distributions', action="store_true", default=False, help="Calculate the dataset distributions.")
    parser.add_argument('--associations', action="store_true", default=False, help="Calculate the dataset feature associations.")
    parser.add_argument('--scale', default="local", help="The scale at which to calculate the dataset distributions.")
    args = parser.parse_args()

    if args.distributions:
        calculate_dataset_distributions(args.data_folder, args.scale)

    if args.associations:
        calculate_dataset_associations(args.data_folder)