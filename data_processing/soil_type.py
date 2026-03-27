import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal
import os
import rasterio
import numpy as np
import argparse
import shapely
gdal.UseExceptions()

def create_soil_type(data_folder, global_folder, scale):
    """
    Create rasters representing the soil classes and soil bulk density, corresponding to the CEMS label rasters.
    """

    # define the metadata and folder locations
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        soil_class_folder = f"{data_folder}/full_subevent/raster_soil_class"
        soil_bulk_density_folder = f"{data_folder}/full_subevent/raster_soil_bulk_density"
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        soil_class_folder = f"{data_folder}/{scale}/soil_class"
        soil_bulk_density_folder = f"{data_folder}/{scale}/soil_bulk_density"
    global_soil_classes_path = f"{global_folder}/global_soil_classes.tif"
    global_soil_bulk_density_path = f"{global_folder}/global_soil_bulk_density.tif"
    if not os.path.isdir(soil_class_folder):
        os.mkdir(soil_class_folder)
    if not os.path.isdir(soil_bulk_density_folder):
        os.mkdir(soil_bulk_density_folder)

    for index in tqdm(range(len(raster_extents)), f"Create soil type rasters for {scale} scale"):

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
            
            if scale == "local":
                soil_type_raster_path = f"{raster_folder}/{subevent}.tif"
            else:  # if scale == "context" or scale == "basin"
                patch_name = raster_extents["patch"].iloc[index]
                soil_type_raster_path = f"{raster_folder}/{patch_name}"
            
            if os.path.isfile(soil_type_raster_path):
                continue

            # extract the given raster extent from the global raster
            resample_alg = "bilinear" if layer_name=="bulk_density" else "nearest"
            # for the bulk density data, impute the missing values using gdal_fillnodata
            if layer_name == "bulk_density":
                if scale == "local":
                    gdal.Warp(soil_type_raster_path, global_path, 
                            srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                            resampleAlg=resample_alg, outputBounds=bounds)
                else:
                    gdal.Warp(soil_type_raster_path, global_path, 
                            srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                            resampleAlg=resample_alg, width=width, height=height, outputBounds=bounds)
                os.system(f"gdal_fillnodata -md=0 -q {soil_type_raster_path} {soil_type_raster_path}")
                if scale == "local":
                    gdal.Warp(soil_type_raster_path, soil_type_raster_path, 
                                srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                                resampleAlg=resample_alg, width=width, height=height, outputBounds=bounds)
            else:
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
            if scale == "local":
                soil_type_raster = np.where(reference_label == 0, 0, soil_type_raster)

            # save the final raster
            with rasterio.open(soil_type_raster_path, "w", **meta, compress="LZW") as file:
                file.write(soil_type_raster, 1)
                file.set_band_description(1, f"soil_{layer_name}")
                file.nodata = None

            if os.path.exists(f"{soil_type_raster_path}.aux.xml"):
                os.remove(f"{soil_type_raster_path}.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create rasters for soil classes and soil bulk density data.")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, context, or basin.")

    args = parser.parse_args()

    create_soil_type(args.data_folder, args.global_folder, args.scale)
