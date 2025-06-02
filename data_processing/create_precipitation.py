import earthaccess
import geopandas as gpd
import os
import pandas as pd
from osgeo import gdal
gdal.UseExceptions()
import rasterio
from rasterio.enums import Resampling
import numpy as np
import time
import argparse
from tqdm import tqdm

def download_precipitation(data_folder):

    # set up and import metadata
    earthaccess.login(strategy="environment")
    global_precipitation_folder = f"{data_folder}/global_precipitation"
    if not os.path.isdir(global_precipitation_folder):
        os.mkdir(global_precipitation_folder)
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")

    # create a list of all the dates that data will need be downloaded for, for the whole dataset
    all_dates = []
    for index in range(len(raster_extents)):
        all_dates += list(pd.date_range(end=raster_extents["date"].dt.date[index], periods=42).date)
    all_dates = list(set(all_dates))
    all_dates.sort()

    all_dates=['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10', '2025-01-11', '2025-01-12', '2025-01-13', '2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19', '2025-01-20', '2025-01-21', '2025-01-22', '2025-01-23', '2025-01-24', '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04', '2025-02-05']

    # download the precipitation data for each date
    for date in tqdm(all_dates):

        geotiff_path = f"{global_precipitation_folder}/{date}_global.tif"
        if os.path.isfile(geotiff_path):
            continue
        print("\n", "\n", geotiff_path)

        # from the API download global precipitation data for each date
        for attempt in range(1, 11):
            try:
                response = earthaccess.search_data(
                short_name="GPM_3IMERGDF",
                temporal=(date, date),
                version="07",
                downloadable=True)
                break
            except RuntimeError as error:
                if attempt == 10:
                    raise
                time.sleep(10)
        if len(response) != 1:
            print(f"{len(response)} responses for date {date}!")
            continue
        file_path = earthaccess.download(response, "data")[0]

        # convert the downloaded file to geotiff
        gdal.Translate(
            destName=geotiff_path,
            srcDS=f'HDF5:"{file_path}"://precipitation',
            format='GTiff', outputSRS='EPSG:6933',
            outputBounds=(-17367530.45, 7314540.83, 17367530.45, -7314540.83))
        
        # project the geotiff to WGS84
        gdal.Warp(geotiff_path, geotiff_path, srcSRS="EPSG:6933", dstSRS="EPSG:4326", format='GTiff', resampleAlg="bilinear")
        
        # replace the the nodata value and update metadata
        with rasterio.open(geotiff_path) as precipitation_file:
            full_precipitation = precipitation_file.read(1)
            meta = precipitation_file.meta.copy()
        full_precipitation = np.where(full_precipitation == -9999.9, 0, full_precipitation)
        meta.update({"driver": "GTiff",
                     "dtype": "float32",
                     "resampling": Resampling.bilinear})
        with rasterio.open(geotiff_path, "w", **meta, compress="LZW") as file:
            file.write(full_precipitation, 1)

        os.remove(file_path)
        time.sleep(10)

def create_precipitation_rasters(data_folder):

    # set up and import metadata
    precipitation_folder = f"{data_folder}/precipitation"
    if not os.path.isdir(precipitation_folder):
        os.mkdir(precipitation_folder)
    precipitation_folder = f"{data_folder}/precipitation"
    global_precipitation_folder = f"{data_folder}/global_precipitation"
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    
    for index in tqdm(range(len(raster_extents))):

        # extract metadata on the subevent
        date = raster_extents["date"].dt.date[index]
        bounds = raster_extents["geometry"][index].bounds
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])
        subevent = raster_extents["subevent"][index]
        file_name = f"{precipitation_folder}/{subevent}.tif"
        num_of_bands = 16

        # determine the dates preceding the subevent and the associated global precipitation rasters for those dates
        precipitation_dates = list(pd.date_range(end=date, periods=42).date)
        global_precipitation_paths = [f"{global_precipitation_folder}/{precipitation_date}_global.tif" for precipitation_date in precipitation_dates]

        # split the dates into 14 day groups and access the precipitation files
        days_1_14 = [rasterio.open(path) for path in global_precipitation_paths[28:42]]
        days_15_28 = [rasterio.open(path) for path in global_precipitation_paths[14:28]]
        days_29_42 = [rasterio.open(path) for path in global_precipitation_paths[0:14]]
        
        # aggregate the first 28 days into two 14 days sums
        sum_days_15_28 = np.sum([file.read(1) for file in days_15_28], axis=0)
        sum_days_29_42 = np.sum([file.read(1) for file in days_29_42], axis=0)
        
        # update the metadata
        meta = days_29_42[0].meta.copy()
        meta.update({"count": num_of_bands})
        meta.pop("nodata", None)

        # create a geotiff with all of the precipitation bands (days)
        with rasterio.open(file_name, "w", **meta, compress="LZW") as file:
            for index in range(14):
                file.write(days_1_14[index].read(1), index+1)
            file.write(sum_days_15_28, 15)
            file.write(sum_days_29_42, 16)
            file.nodata = None

        # match the precipitation geotiff to the subevent raster's extent
        gdal.Warp(file_name, file_name, format='GTiff',
                  resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)
            
        # import the label raster with aois to match the precipitation raster to
        with rasterio.open(f"{data_folder}/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # import the precipitation data
        with rasterio.open(file_name) as precipitation_file:
            precipitation_layers = [precipitation_file.read(index+1) for index in range(num_of_bands)]
            meta = precipitation_file.meta.copy()

        # remove the data where there is no AOI
        precipitation_layers = [np.where(reference_label == 0, 0, precipitation_layer) for precipitation_layer in precipitation_layers]

        # scale the precipitation data by 10 and convert to uint16
        precipitation_layers = [precipitation_layer*10 for precipitation_layer in precipitation_layers]
        precipitation_layers = [np.round(precipitation_layer).astype(np.uint16) for precipitation_layer in precipitation_layers]
        for precipitation_layer in precipitation_layers:
            if np.max(precipitation_layer) > np.iinfo(np.uint16).max:
                print(f"Subevent {subevent} out of range for uint16")
        meta.update({"dtype": "int16"})

        # save the final precipitation geotiff
        precipitation_descriptions = [f"precipitation_day_{index}" for index in range(1, 15)] + ["precipitation_days_15_28", "precipitation_days_29_42"]
        with rasterio.open(file_name, "w", **meta, compress="LZW") as file:
            for index in range(num_of_bands):
                file.write(precipitation_layers[index], index+1)
                file.set_band_description(index+1, precipitation_descriptions[index])

        # close the open original global precipitation date files
        for group in [days_29_42, days_15_28, days_1_14]:
            for file in group:
                file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download precipitation data and create GeoTIFF rasters.")
    
    parser.add_argument("--data_folder", required=True, help="The path to the data folder.")
    parser.add_argument("--download_precipitation", action="store_true", default=False, help="Download the global precipitation data.")
    parser.add_argument("--create_precipitation_rasters", action="store_true", default=False, help="Create raster files of the precipitation data, matching to the CEMS labels.")

    args = parser.parse_args()

    if args.download_precipitation:
        download_precipitation(args.data_folder)

    if args.create_precipitation_rasters:
        create_precipitation_rasters(args.data_folder)