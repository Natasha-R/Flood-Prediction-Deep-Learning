import earthaccess
import geopandas as gpd
import os
import pandas as pd
from osgeo import gdal
gdal.UseExceptions()
import rasterio
import numpy as np
import time
import argparse
import shapely
from tqdm import tqdm

def download_precipitation(data_folder, global_folder):
    """
    Download the (global) precipitation data corresponding to the date of each of the subevent rasters.
    """

    # set up and import metadata
    earthaccess.login(strategy="environment")
    global_precipitation_folder = f"{global_folder}/global_precipitation"
    if not os.path.isdir(global_precipitation_folder):
        os.mkdir(global_precipitation_folder)
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")

    # create a list of all the dates that data will need be downloaded for, for the whole dataset
    all_dates = []
    for index in range(len(raster_extents)):
        all_dates += list(pd.date_range(end=raster_extents["date"].dt.date[index], periods=42).date)
    all_dates = list(set(all_dates))
    all_dates.sort()

    # download the precipitation data for each date
    for date in tqdm(all_dates, "Download precipitation data"):

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
                time.sleep(1)
        if len(response) != 1:
            print(f"{len(response)} responses for date {date}!")
            continue
        file_path = earthaccess.download(response, global_precipitation_folder)[0]

        # extract precipitation band and pre-process
        src = gdal.Open(f'HDF5:"{file_path}"://precipitation')
        precipitation = src.ReadAsArray()
        precipitation = np.flipud(precipitation.T)
        precipitation = np.where(precipitation < 0, 0, precipitation)
        precipitation = precipitation * 10
        precipitation = np.round(precipitation).astype(np.uint16)

        # save the file as geotiff
        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(geotiff_path, precipitation.shape[1], precipitation.shape[0], 1, gdal.GDT_UInt16, options=["COMPRESS=LZW"])
        dst.SetGeoTransform((-180.0, 0.1, 0, 90.0, 0, -0.1))
        dst.SetProjection("EPSG:4326")
        dst.GetRasterBand(1).WriteArray(precipitation)
        dst, src = None, None # close the open files
        os.remove(file_path)

        time.sleep(1)

def create_precipitation_rasters(data_folder, global_folder, scale):
    """
    From the downloaded global precipitation data, create precipitation rasters corresponding to the CEMS label rasters.
    """

    # set up and import metadata
    if scale == "local":
        precipitation_folder = f"{data_folder}/full_subevent/raster_precipitation"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        precipitation_folder = f"{data_folder}/{scale}/precipitation"
    global_precipitation_folder = f"{global_folder}/global_precipitation"
    if not os.path.isdir(precipitation_folder):
        os.mkdir(precipitation_folder)

    for index in tqdm(range(len(raster_extents)), f"Create precipitation rasters for {scale} scale"):

        # extract metadata on the subevent
        date = raster_extents["date"].dt.date[index]
        bounds = raster_extents["geometry"][index].bounds
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])
        subevent = raster_extents["subevent"][index]

        if scale == "local":
            file_name = f"{precipitation_folder}/{subevent}.tif"
        else:
            patch = raster_extents["patch"].iloc[index]
            file_name = f"{precipitation_folder}/{patch}"

        if os.path.isfile(file_name):
            continue

        # determine the dates preceding the subevent and the associated global precipitation rasters for those dates
        precipitation_dates = list(pd.date_range(end=date, periods=42).date)
        global_precipitation_paths = [f"{global_precipitation_folder}/{precipitation_date}_global.tif" for precipitation_date in reversed(precipitation_dates)]

        # split the dates into 14 day groups and access the precipitation files
        days_1_14 = [rasterio.open(path) for path in global_precipitation_paths[0:14]]
        days_15_28 = [rasterio.open(path) for path in global_precipitation_paths[14:28]]
        days_29_42 = [rasterio.open(path) for path in global_precipitation_paths[28:42]]
        
        # aggregate the first 28 days into two 14 days sums
        sum_days_15_28 = np.sum([file.read(1) for file in days_15_28], axis=0)
        sum_days_29_42 = np.sum([file.read(1) for file in days_29_42], axis=0)
        
        # create a geotiff with all of the (global) precipitation bands (days)
        meta = days_29_42[0].meta.copy()
        meta.update({"count": 16})
        precipitation_descriptions = [f"precipitation_day_{index}" for index in range(14)] + ["precipitation_days_14_27", "precipitation_days_28_41"]
        with rasterio.open(file_name, "w", **meta, compress="LZW") as file:
            for index in range(14):
                file.write(days_1_14[index].read(1), index+1)
            file.write(sum_days_15_28, 15)
            file.write(sum_days_29_42, 16)
            file.nodata = None
            for index in range(16):
                file.set_band_description(index+1, precipitation_descriptions[index])

        # match the precipitation geotiff to the subevent raster's extent
        gdal.Warp(file_name, file_name, format='GTiff', creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"],
                resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)

        if scale == "local": # remove the data where there is no AOI
            with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
                reference_label = reference_file.read(1)
            with rasterio.open(file_name) as precipitation_file:
                precipitation_layers = [precipitation_file.read(index+1) for index in range(16)]
                meta = precipitation_file.meta.copy()
            precipitation_layers = [np.where(reference_label == 0, 0, precipitation_layer) for precipitation_layer in precipitation_layers]
            with rasterio.open(file_name, "w", **meta, compress="LZW") as file:
                for index in range(16):
                    file.write(precipitation_layers[index], index+1)
                    file.set_band_description(index+1, precipitation_descriptions[index])

        # close the open original global precipitation date files
        for group in [days_29_42, days_15_28, days_1_14]:
            for file in group:
                file.close()
        if os.path.exists(f"{file_name}.aux.xml"):
            os.remove(f"{file_name}.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download precipitation data and create GeoTIFF rasters.")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--download_precipitation", action="store_true", default=False, help="Download the global precipitation data.")
    parser.add_argument("--create_precipitation_rasters", action="store_true", default=False, help="Create raster files of the precipitation data, matching to the CEMS labels.")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, nearby, context, con_context, or basin.")

    args = parser.parse_args()

    if args.download_precipitation:
        download_precipitation(args.data_folder, args.global_folder)

    if args.create_precipitation_rasters:
        create_precipitation_rasters(args.data_folder, args.global_folder, args.scale)