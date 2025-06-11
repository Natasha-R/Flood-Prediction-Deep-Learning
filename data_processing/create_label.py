import argparse
import osmnx as ox
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import time
import os
import rasterio
import numpy as np
from osgeo import gdal
import wget
import shutil
gdal.UseExceptions()

def create_permanent_water_geojson(data_folder, global_folder):
    """
    Create geojsons containing polygons corresponding to all permanent waters and seas (as classified by OSM).
    """

    # import the seas and oceans polygons
    print("Importing in all seas and oceans polygons...")
    seas = gpd.read_file(f"{global_folder}/global_seas_polygons/water_polygons.shp")
    print("Import complete")

    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")

    permanent_water_geojson_folder = f"{data_folder}/full_subevent/geojson_permanent_water"
    if not os.path.isdir(permanent_water_geojson_folder):
        os.mkdir(permanent_water_geojson_folder)

    for index in tqdm(range(len(raster_extents))):

        path = f"{permanent_water_geojson_folder}/{raster_extents['subevent'][index]}.geojson"
        if os.path.isfile(path):
            continue

        # find the extent of the raster
        geometry = raster_extents["geometry"][index].bounds

        # return all water polygons and lines from OSM that fall within the raster extent
        tags = {
            "natural": ["water"],
            "waterway": True,
            "landuse": ["reservoir"]
            }
        water_polygons = ox.features_from_bbox(bbox=geometry, tags=tags).reset_index()
        water_polygons = water_polygons[(water_polygons["element"] == "way") | (water_polygons["element"] == "relation")][["geometry"]].reset_index(drop=True)

        # return all seas and oceans polygons that fall within the raster extent
        seas_polygons = seas.iloc[list(seas.sindex.intersection(geometry))]

        # merge all sea and water polygons and then clip to the extent of the raster
        permanent_water = pd.concat([seas_polygons, water_polygons], ignore_index=True)[["geometry"]]
        permanent_water = gpd.clip(permanent_water, raster_extents.iloc[[index]])

        # convert to a coordinate system that uses metres, and save as geojson
        crs = gpd.read_file(f"{'/'.join(permanent_water_geojson_folder.split('/')[:-1])}/geojson_labels/{raster_extents["subevent"][index]}.geojson").crs
        permanent_water = permanent_water.to_crs(crs)
        permanent_water.to_file(path)

        time.sleep(1)

def create_permanent_water_raster(data_folder):
    """
    Convert the geojsons of permanent waters and seas into raster format.
    """

    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    permanent_water_geojson_folder = f"{data_folder}/full_subevent/geojson_permanent_water"
    permanent_water_raster_folder = f"{data_folder}/full_subevent/raster_permanent_water"
    if not os.path.isdir(permanent_water_raster_folder):
        os.mkdir(permanent_water_raster_folder)

    for index in tqdm(range(len(raster_extents))):

        # import the data and extract the metadata
        path = f"{permanent_water_geojson_folder}/{raster_extents['subevent'][index]}.geojson"
        permanent_water = gpd.read_file(path)
        subevent = raster_extents["subevent"][index]
        utm_raster_extent = raster_extents["geometry"].to_crs(permanent_water.crs)[index].bounds
        wgs84_raster_extent = raster_extents["geometry"][index].bounds

        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
             reference_label = reference_file.read(1)
             height, width = reference_label.shape
             meta = reference_file.meta.copy()
             meta.pop("nodata", None)

        # rasterize the permanent water polygons and match to the cems label raster extent
        gdal.Rasterize(f"{permanent_water_raster_folder}/{subevent}_utm.tif", path, 
                       format="GTiff", xRes=10, yRes=10, burnValues=[1.0], resampleAlg="nearest", outputBounds=utm_raster_extent)
        gdal.Warp(f"{permanent_water_raster_folder}/{subevent}_wgs84.tif", f"{permanent_water_raster_folder}/{subevent}_utm.tif", 
                  srcSRS=permanent_water.crs, dstSRS="EPSG:4326", width=width, height=height, format="GTiff", outputBounds=wgs84_raster_extent,
                  resampleAlg="nearest", outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"])

        # remove all of the permanent water data from outside of the label AOIs and save as raster
        with rasterio.open(f"{permanent_water_raster_folder}/{subevent}_wgs84.tif") as perm_water_file:
            perm_water_raster = perm_water_file.read(1)
        perm_water_masked = np.where(reference_label == 0, 0, perm_water_raster)
        with rasterio.open(f"{permanent_water_raster_folder}/{subevent}.tif", "w", **meta, compress="LZW") as file:
            file.write(perm_water_masked, 1)
            file.nodata = None

        # delete the temporary intermediary files
        os.remove(f"{permanent_water_raster_folder}/{subevent}_utm.tif")
        os.remove(f"{permanent_water_raster_folder}/{subevent}_wgs84.tif")

def combine_cems_and_permanent_water(data_folder):
    """
    Subtract the permanent water from the flood labels in the CEMS rasters, to create the final label raster.
    """

    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")

    for index in tqdm(range(len(raster_extents))):
        
        # load in the rasters
        subevent = raster_extents["subevent"][index]
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as cems_file:
            cems_raster = cems_file.read(1)
            meta = cems_file.meta.copy()
            meta.pop("nodata", None)
        with rasterio.open(f"{data_folder}/full_subevent/raster_permanent_water/{subevent}.tif") as perm_water_file:
            perm_water_raster = perm_water_file.read(1)

        # remove all flood labels from locations of permanent water
        label_raster = np.where(perm_water_raster == 1, 1, cems_raster)

        # save the final data as a raster
        label_raster_folder = f"{data_folder}/full_subevent/raster_label"
        if not os.path.isdir(label_raster):
            os.mkdir(label_raster)
        with rasterio.open(f"{label_raster_folder}/{subevent}.tif", "w", **meta, compress="LZW") as file:
            file.write(label_raster, 1)
            file.nodata = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate GeoJSON and raster files comprising the permanent water and combine them with the CEMS labels.")

    parser.add_argument("--data_folder", required=True, help="The path to the data folder.")
    parser.add_argument("--global_folder", default=None, help="The path to the folder containing the global data")
    parser.add_argument("--create_permanent_water_geojson", action="store_true", default=False, help="Create the permanent water GeoJSON files.")
    parser.add_argument("--create_permanent_water_raster", action="store_true", default=False, help="Create the permanent water rasters from the GeoJSON files.")
    parser.add_argument("--combine_cems_and_permanent_water", action="store_true", default=False, help="Combine the permanent water and CEMS data to create the labels.")

    args = parser.parse_args()

    if args.create_permanent_water_geojson:
        create_permanent_water_geojson(args.data_folder, args.global_folder)

    if args.create_permanent_water_raster:
        create_permanent_water_raster(args.data_folder)

    if args.combine_cems_and_permanent_water:
        combine_cems_and_permanent_water(args.data_folder)