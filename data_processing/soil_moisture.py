import earthaccess
import rasterio
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from osgeo import gdal
import os
from rasterio.enums import Resampling
import time
import pandas as pd
import argparse
import shapely
gdal.UseExceptions()

def download_soil_moisture_data(data_folder, global_folder):
    """
    Download the (global) soil moisture data corresponding to the date of each of the subevent rasters.
    """

    earthaccess.login(strategy="environment")

    # import the metadata and find all of the unique dates for which to download data for
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    raster_extents["one_day_previous"] = raster_extents["date"] - pd.Timedelta(days=1)
    raster_extents["one_week_previous"] = raster_extents["date"] - pd.Timedelta(days=7)
    all_dates = list(set(list(raster_extents["one_day_previous"]) + list(raster_extents["one_week_previous"])))

    global_soil_moisture_folder = f"{global_folder}/global_soil_moisture/"
    if not os.path.isdir(global_soil_moisture_folder):
        os.mkdir(global_soil_moisture_folder)

    for date in tqdm(all_dates):

        date = date.date()
        if (os.path.isfile(f"{global_soil_moisture_folder}/{date}_sm_surface_global.tif") 
            and os.path.isfile(f"{global_soil_moisture_folder}/{date}_sm_rootzone_global.tif")):
            continue

        # download the soil moisture data from the NASA Earth Science Data API
        print("\n", "\n", f"{global_soil_moisture_folder}/{date}.tif", flush=True)
        response = earthaccess.search_data(
            short_name="SPL4SMGP",
            temporal=(f"{date}T12:00:00Z", f"{date}T15:00:00Z", True),
            version="008",
            downloadable=True)
        if len(response) != 1:
            print(f"{len(response)} responses for date {date}")
            response = [response[0]]
        file_path = earthaccess.download(response, global_soil_moisture_folder)[0]

        # convert the global hdf5 file into GeoTIFFs with the correct projection
        for layer in ["sm_surface", "sm_rootzone"]:
            gdal.Translate(destName=f"{global_soil_moisture_folder}/{date}_{layer}_global.tif",
                            srcDS=f'HDF5:"{file_path}"://Geophysical_Data/{layer}',
                            format='GTiff', outputSRS='EPSG:6933', creationOptions=["COMPRESS=LZW"],
                            outputBounds=(-17367530.45, 7314540.83, 17367530.45, -7314540.83))

        os.remove(file_path)
        time.sleep(10)

def create_soil_moisture_rasters(data_folder, global_folder, scale):
    """
    From the downloaded global soil moisture data, create precipitation rasters corresponding to the CEMS label rasters.
    """

    # import the metadata
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        soil_moisture_folders = [f"{data_folder}/full_subevent/raster_soil_moisture_one_day/",
                                f"{data_folder}/full_subevent/raster_soil_moisture_one_week/"]
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        soil_moisture_folders = [f"{data_folder}/{scale}/soil_moisture_one_day/",
                                 f"{data_folder}/{scale}/soil_moisture_one_week/"]
    for soil_moisture_folder in soil_moisture_folders:
        if not os.path.isdir(soil_moisture_folder):
            os.mkdir(soil_moisture_folder)
    raster_extents["one_day"] = raster_extents["date"] - pd.Timedelta(days=1)
    raster_extents["one_week"] = raster_extents["date"] - pd.Timedelta(days=7)

    for index in tqdm(range(len(raster_extents))):

        # extract data for both one day previous to the subevent, and one week before
        for time_frame in ["one_day", "one_week"]:
            
            # extract out the metadata for the subevent raster
            date = raster_extents[time_frame].dt.date[index]
            bounds = raster_extents["geometry"][index].bounds
            height = int(raster_extents["height"].iloc[index])
            width = int(raster_extents["width"].iloc[index])
            subevent = raster_extents["subevent"][index]
            
            if scale == "local":
                # import the label raster to match the soil moisture raster to
                with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
                    reference_label = reference_file.read(1)
                save_path = f"{data_folder}/full_subevent/raster_soil_moisture_{time_frame}/{subevent}"
            else:  # if scale == "context" or scale == "basin"
                patch_name = raster_extents["patch"].iloc[index][:-4]
                save_path = f"{data_folder}/{scale}/soil_moisture_{time_frame}/{patch_name}"

            # extract the surface and rootzone soil moisture layers
            soil_moistures = []
            for layer in ["sm_surface", "sm_rootzone"]:
                
                # extract only the raster area from the corresponding global GeoTIFF and convert to WGS84
                gdal.Warp(f"{save_path}_{layer}.tif", f"{global_folder}/global_soil_moisture/{date}_{layer}_global.tif", 
                            srcSRS="EPSG:6933", dstSRS="EPSG:4326", format='GTiff',
                            resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)

                # import the soil moisture raster data and metadata
                with rasterio.open(f"{save_path}_{layer}.tif") as soil_moisture_file:
                    soil_moisture = soil_moisture_file.read(1)
                    meta = soil_moisture_file.meta.copy()

                # keep only soil moisture data within the AOI bounds, and scale data by 10,000 to store as uint16
                if scale == "local":
                    soil_moisture = np.where(reference_label == 0, 0, soil_moisture)
                soil_moisture = np.where(soil_moisture == -9999, 0, soil_moisture)
                soil_moisture = soil_moisture*10000
                soil_moisture = np.round(soil_moisture).astype(np.uint16)
                soil_moistures.append(soil_moisture)

                # remove the intermediary files
                os.remove(f"{save_path}_{layer}.tif")
            
            # save the two soil moisture bands to a GeoTIFF
            meta.update({"driver": "GTiff",
                        "dtype": "uint16",
                        "resampling": Resampling.bilinear,
                        "count": 2})
            meta.pop("nodata", None)
            with rasterio.open(f"{save_path}.tif", "w", **meta, compress="LZW") as file:
                file.write(soil_moistures[0], 1)
                file.set_band_description(1, "soil_moisture_surface")
                file.write(soil_moistures[1], 2)
                file.set_band_description(2, "soil_moisture_rootzone")
                file.nodata = None

            os.remove(f"{save_path}.tif.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create raster files representing the soil moisture values.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--download_soil_moisture_data", action="store_true", default=False, help="Download the global soil moisture data.")
    parser.add_argument("--create_soil_moisture_rasters", action="store_true", default=False, help="Create raster files of the soil moisture data, matching to the CEMS labels.")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, context, or basin.")

    args = parser.parse_args()

    if args.download_soil_moisture_data:
        download_soil_moisture_data(args.data_folder, args.global_folder)

    if args.create_soil_moisture_rasters:
        create_soil_moisture_rasters(args.data_folder, args.global_folder, args.scale)