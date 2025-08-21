import geopandas as gpd
from osgeo import gdal
import numpy as np
import rasterio
from rasterio.windows import Window
import json
from tqdm import tqdm
import os
import argparse
import warnings
import pandas as pd
import shapely
from shapely.geometry import box
gdal.UseExceptions()
pd.options.mode.chained_assignment = None

def create_label_local_patches(data_folder):
    """
    Divide the label rasters into 256x256 patches.
    """

    # read in the metadata
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    subevent_patches = {}
    if not os.path.isdir(f"{data_folder}/local"):
        os.mkdir(f"{data_folder}/local/")
    if not os.path.isdir(f"{data_folder}/local/label/"):
        os.mkdir(f"{data_folder}/local/label/")

    for index in tqdm(range(len(raster_extents))):

        selected_patches = []
        subevent = raster_extents["subevent"][index]

        # extend the label raster by at least 256x256 pixels, to ensure that all label data is included in a patch
        new_bounds = list(raster_extents["geometry"][index].bounds)
        new_bounds[1] -= 0.03
        new_bounds[2] += 0.07
        gdal.Warp(destNameOrDestDS=f"{data_folder}/local/label/{subevent}_padded.tif", 
                  srcDSOrSrcDSTab=f"{data_folder}/full_subevent/raster_label/{subevent}.tif", 
                  resampleAlg="nearest", outputBounds=new_bounds)

        with rasterio.open(f"{data_folder}/local/label/{subevent}_padded.tif") as full_subevent_file:

            # calculate the number of patches in the full raster
            width = full_subevent_file.width
            height = full_subevent_file.height
            meta = full_subevent_file.meta.copy()
            patches_per_column = width // 256
            patches_per_row = height // 256
            num_patches = patches_per_column * patches_per_row

            for patch_index in range(0, num_patches):

                # extract out a 256x256 patch at each index
                patch_row = patch_index % patches_per_column
                patch_col = patch_index // patches_per_column
                row_pixel_start = patch_row * 256
                col_pixel_start = patch_col * 256
                window = Window(row_pixel_start, col_pixel_start, 256, 256)
                patch = full_subevent_file.read(window=window)

                # if there is no data (aoi) in the patch, then discard it
                if np.all(patch == 0):
                    continue
                else: 
                    selected_patches.append(patch_index)

                # save the new patch in the local/label folder
                meta.update({"transform": full_subevent_file.window_transform(window),
                            "height": 256, "width": 256})
                with rasterio.open(f"{data_folder}/local/label/{subevent}_{patch_index:06}.tif", "w", **meta, compress="LZW") as file:
                    file.write(patch)

        # save the indices of the selected patches, so they can be utilised extracting patches from the other data features
        subevent_patches[subevent] = {"selected_patches": selected_patches, 
                                      "num_patches": num_patches,
                                      "patches_per_column": patches_per_column,
                                      "patches_per_row": patches_per_row}
        os.remove(f"{data_folder}/local/label/{subevent}_padded.tif")
    
    json.dump(subevent_patches, open(f"{data_folder}/metadata/subevent_patches.json", "w"))

def create_features_local_patches(data_folder):
    """
    Divide all of the data input features into 256x256 patches, matching to the label patches.
    """

    # import the metadata
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    subevent_patches = json.load(open(f"{data_folder}/metadata/subevent_patches.json"))
    features = ["precipitation", "dem", "permanent_water", "sentinel1", "sentinel2", "soil_moisture_one_week", "soil_moisture_one_day", "soil_class", "soil_bulk_density", "flow_accumulation"]
    resample_alg = {"precipitation": "bilinear", "dem": "bilinear", "sentinel1": "bilinear", "sentinel2": "bilinear", "flow_accumulation": "bilinear",
                    "soil_moisture_one_week": "bilinear", "soil_moisture_one_day": "bilinear", "soil_bulk_density": "bilinear",
                    "soil_class": "nearest", "permanent_water": "nearest"}
    for feature in features:
        if not os.path.isdir(f"{data_folder}/local/{feature}/"):
            os.mkdir(f"{data_folder}/local/{feature}/")

    for feature in tqdm(features, desc="feature"):

        for index in tqdm(range(len(raster_extents)), desc="raster", leave=False):

            # import metadata on the indices to extract from this raster, calculated from the label data
            subevent = raster_extents["subevent"][index]
            selected_patches = subevent_patches[subevent]["selected_patches"]
            patches_per_column = subevent_patches[subevent]["patches_per_column"]

            # extend the raster by at least 256x256 pixels, to ensure that all data is included in a patch
            new_bounds = list(raster_extents["geometry"][index].bounds)
            new_bounds[1] -= 0.03
            new_bounds[2] += 0.07
            gdal.Warp(destNameOrDestDS=f"{data_folder}/local/{feature}/{subevent}_padded.tif", 
                    srcDSOrSrcDSTab=f"{data_folder}/full_subevent/raster_{feature}/{subevent}.tif", 
                    resampleAlg=resample_alg[feature], outputBounds=new_bounds)

            # extract out patches from the full subevent file for the given feature
            with rasterio.open(f"{data_folder}/local/{feature}/{subevent}_padded.tif") as full_subevent_file:
                meta = full_subevent_file.meta.copy()
                descriptions = full_subevent_file.descriptions

                for patch_index in selected_patches:

                    # select patches with indices selected from the label data
                    patch_row = patch_index % patches_per_column
                    patch_col = patch_index // patches_per_column
                    row_pixel_start = patch_row * 256
                    col_pixel_start = patch_col * 256
                    window = Window(row_pixel_start, col_pixel_start, 256, 256)
                    patch = full_subevent_file.read(window=window)

                    # save the patch
                    meta.update({"transform": full_subevent_file.window_transform(window),
                                 "height": 256, "width": 256})
                    with rasterio.open(f"{data_folder}/local/{feature}/{subevent}_{patch_index:06}.tif", "w", **meta, compress="LZW") as file:
                        file.write(patch)
                        for i, description in enumerate(descriptions, start=1):
                            if description is not None:
                                file.set_band_description(i, description)

            os.remove(f"{data_folder}/local/{feature}/{subevent}_padded.tif")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Divide the subevent rasters into 256x256 patches to form the local dataset.")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--create_label_local_patches", action="store_true", default=False, help="Divide the label rasters into 256x256 patches.")
    parser.add_argument("--create_features_local_patches", action="store_true", default=False, help="Divide all of the data input features into 256x256 patches, matching to the label patches.")

    args = parser.parse_args()

    if args.create_label_local_patches:
        create_label_local_patches(args.data_folder)
    
    if args.create_features_local_patches:
        create_features_local_patches(args.data_folder)