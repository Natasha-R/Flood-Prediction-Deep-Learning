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
gdal.UseExceptions()

def create_permanent_water_geojson(perm_water_geojson_folder, seas_polygons_path):

    # import in metadata and the seas and oceans polygons
    raster_extents = gpd.read_file("metadata/raster_extent.geojson")
    print("Importing in all seas and oceans polygons...")
    seas = gpd.read_file(seas_polygons_path)
    print("Import complete")

    for index in tqdm(range(len(raster_extents))):

        path = f"{perm_water_geojson_folder}/{raster_extents['subevent'][index]}.geojson"
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
        crs = gpd.read_file(f"{'/'.join(perm_water_geojson_folder.split('/')[:-1])}/geojson_labels/{raster_extents["subevent"][index]}.geojson").crs
        permanent_water = permanent_water.to_crs(crs)
        permanent_water.to_file(path)

        time.sleep(1)

def create_permanent_water_raster(perm_water_geojson_folder, perm_water_raster_folder, cems_raster_folder):

    raster_extents = gpd.read_file("metadata/raster_extent.geojson")

    for index in tqdm(range(len(raster_extents))):

        # import the data and extract the metadata
        path = f"{perm_water_geojson_folder}/{raster_extents['subevent'][index]}.geojson"
        permanent_water = gpd.read_file(path)
        subevent = raster_extents["subevent"][index]
        utm_raster_extent = raster_extents["geometry"].to_crs(permanent_water.crs)[index].bounds
        wgs84_raster_extent = raster_extents["geometry"][index].bounds

        with rasterio.open(f"{cems_raster_folder}/{subevent}.tif") as reference_file:
             reference_label = reference_file.read(1)
             height, width = reference_label.shape
             meta = reference_file.meta.copy()

        # rasterize the permanent water polygons and match to the cems label raster extent
        gdal.Rasterize(f"{perm_water_raster_folder}/{subevent}_utm.tif", path, 
                       format="GTiff", xRes=10, yRes=10, burnValues=[1.0], outputBounds=utm_raster_extent)
        gdal.Warp(f"{perm_water_raster_folder}/{subevent}_wgs84.tif", f"{perm_water_raster_folder}/{subevent}_utm.tif", 
                  srcSRS=permanent_water.crs, dstSRS="EPSG:4326", width=width, height=height, format="GTiff", outputBounds=wgs84_raster_extent,
                  outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"])

        # remove all of the permanent water data from outside of the label AOIs and save as raster
        with rasterio.open(f"{perm_water_raster_folder}/{subevent}_wgs84.tif") as perm_water_file:
            perm_water_raster = perm_water_file.read(1)
        perm_water_masked = np.where(reference_label == 0, 0, perm_water_raster)
        with rasterio.open(f"{perm_water_raster_folder}/{subevent}.tif", "w", **meta, compress="LZW") as file:
            file.write(perm_water_masked, 1)

        # delete the temporary intermediary files
        os.remove(f"{perm_water_raster_folder}/{subevent}_utm.tif")
        os.remove(f"{perm_water_raster_folder}/{subevent}_wgs84.tif")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate GeoJSON and raster files comprising the permanent water.")

    parser.add_argument("--perm_water_geojson_folder", required=True, help="The path to the permanent water GeoJSON folder.")
    parser.add_argument("--perm_water_raster_folder", default=None, help="The path to the permanent water raster folder.")
    parser.add_argument("--cems_raster_folder", default=None, help="The path to the CEMS raster folder.")
    parser.add_argument("--seas_polygons_path", default=None, help="The path to the seas polygon file.")

    parser.add_argument("--create_geojson", action="store_true", default=False, help="Create the GeoJSON files.")
    parser.add_argument("--create_raster", action="store_true", default=False, help="Create the rasters from the GeoJSON files.")

    args = parser.parse_args()

    if args.create_geojson:
        create_permanent_water_geojson(args.perm_water_geojson_folder, args.seas_polygons_path)

    if args.create_raster:
        create_permanent_water_raster(args.perm_water_geojson_folder, args.perm_water_raster_folder, args.cems_raster_folder)