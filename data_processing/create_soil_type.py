import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal
import os
import rasterio
import numpy as np
import argparse
gdal.UseExceptions()

def create_soil_type(data_folder, global_folder):
    """
    Create rasters representing the soil classes and soil bulk density, corresponding to the CEMS label rasters.
    """

    # define the metadata and folder locations
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    global_soil_classes_path = f"{global_folder}/global_soil_classes.tif"
    global_soil_bulk_density_path = f"{global_folder}/global_soil_bulk_density.tif"
    soil_class_folder = f"{data_folder}/full_subevent/raster_soil_class"
    if not os.path.isdir(soil_class_folder):
        os.mkdir(soil_class_folder)
    soil_bulk_density_folder = f"{data_folder}/full_subevent/raster_soil_bulk_density"
    if not os.path.isdir(soil_bulk_density_folder):
        os.mkdir(soil_bulk_density_folder)

    for index in tqdm(range(len(raster_extents))):

        # import the metadata for the current raster
        bounds = raster_extents["geometry"][index].bounds
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])
        subevent = raster_extents["subevent"][index]
                
        # import the CEMS label as a reference to the extent of the AOIs
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # create files for soil class and soil bulk density
        for layer_name, global_path, raster_folder in zip(["class", "bulk_density"], 
                                                          [global_soil_classes_path, global_soil_bulk_density_path],
                                                          [soil_class_folder, soil_bulk_density_folder]):
            
            soil_type_raster_path = f"{raster_folder}/{subevent}.tif"
            
            if os.path.isfile(soil_type_raster_path):
                continue

            # extract the given raster extent from the global raster
            resample_alg = "bilinear" if layer_name=="bulk_density" else "nearest"
            gdal.Warp(soil_type_raster_path, global_path, 
                      srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                      resampleAlg=resample_alg, width=width, height=height, outputBounds=bounds)
            
            # keep only data within the AOI bounds and fix the noData value
            with rasterio.open(soil_type_raster_path) as soil_type_file:
                soil_type_raster = soil_type_file.read(1)
                meta = soil_type_file.meta.copy()
                meta.pop("nodata", None)
            if layer_name == "class":
                soil_type_raster = np.where(soil_type_raster == 255, 254, soil_type_raster)
                soil_type_raster = soil_type_raster + 1
                soil_type_raster = np.where(soil_type_raster == 255, 0, soil_type_raster)
            else:
                soil_type_raster = np.where(soil_type_raster == -32768, 0, soil_type_raster)
            soil_type_raster = np.where(reference_label == 0, 0, soil_type_raster)

            # save the final raster
            with rasterio.open(soil_type_raster_path, "w", **meta, compress="LZW") as file:
                file.write(soil_type_raster, 1)
                file.set_band_description(1, f"soil_{layer_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create rasters for soil classes and soil bulk density data.")
    
    parser.add_argument("--data_folder", required=True, help="The path to the data folder.")
    parser.add_argument("--global_folder", required=True, help="The path to the folder containing the global data")

    args = parser.parse_args()

    create_soil_type(args.data_folder, args.global_folder)
