from tqdm import tqdm
import os
import pandas as pd
from functools import reduce
import numpy as np
import tifffile as tf
import argparse

def calculate_dataset_distributions(data_folder):
    """
    Calculate the distributions of each of the features.
    """

    features = [(feature_name, band_index) for feature_name, feature_count in zip(["dem", "permanent_water", "soil_bulk_density", "soil_class", "label", "flow_accumulation",
                                                                                   "soil_moisture_one_day", "soil_moisture_one_week", "sentinel2", "sentinel1"],
                                                                                   [1, 1, 1, 1, 1, 1, 2, 2, 12, 3]) for band_index in range(feature_count)]
    features = features + [("precipitation", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)), ("precipitation", 14), ("precipitation", 15)]
    for feature, band in tqdm(features):
        paths = [path.path for path in os.scandir(f"{data_folder}/full_subevent/raster_{feature}/") if path.path.endswith(".tif")]
        feature_value_counts = []

        for path in paths:
            raster = tf.imread(path)
            reference = tf.imread(f"{data_folder}/full_subevent/raster_cems/{path.split('/')[-1]}")
            if raster.ndim == 2: raster = np.expand_dims(raster, axis=-1)
            raster = raster[:, :, band]
            raster = raster[reference != 0]
            feature_value_counts.append(pd.Series(raster.flatten()).value_counts())

        feature_value_counts = pd.DataFrame(reduce(lambda a, b: a.add(b, fill_value=0), feature_value_counts).astype(int)).reset_index(names="value")
        feature_value_counts["feature"] = f"{feature}_{band}"
        feature_value_counts_path = f"{data_folder}/metadata/dataset_distributions.csv"
        feature_value_counts.to_csv(feature_value_counts_path, mode="a", header=not os.path.exists(feature_value_counts_path), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the distributions of each of the features.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    args = parser.parse_args()

    calculate_dataset_distributions(args.data_folder)