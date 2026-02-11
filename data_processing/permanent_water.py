import argparse
import osmnx as ox
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import time
import os
import shapely
from shapely.geometry import box
import rasterio
import numpy as np
from osgeo import gdal
import datetime
gdal.UseExceptions()

def download_global_permanent_water(data_folder, global_folder):
    """
    Download global permanent water data from OSM.
    """

    # create a global grid of 1x1 degree polygons 
    grid_path = f"{data_folder}/metadata/global_grid_1x1.geojson"
    if not os.path.isfile(grid_path):
        polygons, ids = [], []
        for longitude in range(-180, 180):
            for latitude in range(-90, 90):
                polygons.append(box(longitude, latitude, longitude+1, latitude+1))
                ids.append(f"{latitude}_{longitude}")
        global_grid = gpd.GeoDataFrame({"grid_id": ids, "geometry": polygons}, crs="EPSG:4326")
        global_grid.to_file(f"{global_folder}/global_permanent_water/global_grid_1x1.geojson", driver="GeoJSON")

    global_grid = gpd.read_file(f"{global_folder}/global_permanent_water/global_grid_1x1_reduced.geojson")

    # for each grid tile, download all water polygons and lines from OSM
    tags = {"natural": ["water"],
            "waterway": True,
            "landuse": ["reservoir"]}
    for index in tqdm(range(len(global_grid))):
        time.sleep(10)
        geometry = global_grid["geometry"][index].bounds
        grid_id = global_grid["grid_id"][index]
        try:
            permanent_water = ox.features_from_bbox(bbox=geometry, tags=tags).reset_index()
            permanent_water = permanent_water[(permanent_water["element"] == "way") | (permanent_water["element"] == "relation")][["geometry"]].reset_index(drop=True)
            if not permanent_water.empty:
                permanent_water.to_file(f"{global_folder}/global_permanent_water/{grid_id}.geojson")
                print("\n", datetime.datetime.now(), grid_id, index, flush=True)
        except Exception as exception:
            print(f"\n {grid_id}: {exception} ({datetime.datetime.now()})", flush=True)
            continue

def create_permanent_water_rasters(data_folder, global_folder, scale):
    """
    Create permanent water rasters corresponding to the CEMS labels.
    """

    # import the seas and oceans polygons
    print("Importing in all seas and oceans polygons...")
    seas = gpd.read_file(f"{global_folder}/global_seas_polygons/water_polygons.shp")
    print("Import complete")

    if type(scale) is not list:
        scale = [scale]

    for scale_name in scale:

        # read in metadata
        if scale_name == "local":
            raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
            permanent_water_raster_folder = f"{data_folder}/full_subevent/raster_permanent_water"
        else: # if scale == "context" or scale == "basin"
            raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
            raster_extents["geometry"] = raster_extents[f"{scale_name}_geometry"].apply(shapely.wkt.loads)
            permanent_water_raster_folder = f"{data_folder}/{scale_name}/permanent_water"
            if scale_name == "basin":
                print("Importing in all global rivers polygons...")
                global_rivers = gpd.read_file(f"{global_folder}/global_rivers/HydroRIVERS_v10.shp")
                global_rivers = global_rivers[global_rivers["ORD_CLAS"] == 1]
                print("Import complete")
        global_grid = gpd.read_file(f"{global_folder}/global_permanent_water/global_grid_1x1_reduced.geojson")
        if not os.path.isdir(permanent_water_raster_folder):
            os.mkdir(permanent_water_raster_folder)

        for index in tqdm(range(len(raster_extents))):

            subevent = raster_extents["subevent"][index]
            if scale_name == "local":
                save_path = f"{permanent_water_raster_folder}/{subevent}"
            else: # if scale == "context" or scale == "basin"
                patch_name = raster_extents["patch"].iloc[index]
                save_path = f"{permanent_water_raster_folder}/{patch_name[:-4]}"
            if os.path.isfile(save_path + ".tif"):
                continue

            geometry = raster_extents["geometry"][index].bounds

            # return all seas and oceans polygons that fall within the raster extent
            seas_polygons = seas.iloc[list(seas.sindex.intersection(geometry))]

            # extract all water polygons from the intersecting permanent water grid tiles
            if scale_name == "basin":
                water_polygons = global_rivers[global_rivers.geometry.intersects(raster_extents["geometry"][index])]
            else:
                grid_intersects = global_grid[global_grid.intersects(raster_extents["geometry"][index])]
                grid_paths = [f"{global_folder}/global_permanent_water/{grid_id}.geojson" for grid_id in list(grid_intersects["grid_id"])]
                grid_tiles = [gpd.read_file(grid_path) for grid_path in grid_paths if os.path.isfile(grid_path)]
                water_polygons = pd.concat(grid_tiles).drop_duplicates("geometry") if grid_tiles else gpd.GeoDataFrame(columns=["geometry"])

            # merge all sea and water polygons and then clip to the extent of the raster
            permanent_water = pd.concat([seas_polygons, water_polygons], ignore_index=True)[["geometry"]]
            permanent_water = gpd.clip(permanent_water, raster_extents.iloc[[index]])

            # convert to a coordinate system that uses metres, and save as geojson
            crs = gpd.read_file(f"{data_folder}/full_subevent/geojson_cems/{subevent}.geojson").crs
            permanent_water = permanent_water.to_crs(crs)
            permanent_water.to_file(save_path + ".geojson")

            # find the extents of the raster in different coordinate systems
            utm_raster_extent = raster_extents["geometry"].to_crs(permanent_water.crs)[index].bounds
            wgs84_raster_extent = raster_extents["geometry"][index].bounds

            if scale_name=="local":
                # use the label as the reference to create the permanent water raster
                with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
                    reference_label = reference_file.read(1)
                    height, width = reference_label.shape
                    meta = reference_file.meta.copy()
                    meta.pop("nodata", None)
            else:
                height, width = 256, 256

            # rasterize the permanent water polygons and match to the cems label raster extent
            if scale_name == "local":
                gdal.Rasterize(f"{save_path}_utm.tif", save_path + ".geojson", 
                            format="GTiff", xRes=10, yRes=10, burnValues=[1.0], outputBounds=utm_raster_extent)
            else:
                gdal.Rasterize(f"{save_path}_utm.tif", save_path + ".geojson", 
                               format="GTiff", height=256, width=256, burnValues=[1.0], outputBounds=utm_raster_extent)
            gdal.Warp(f"{save_path}_wgs84.tif", f"{save_path}_utm.tif", 
                    srcSRS=permanent_water.crs, dstSRS="EPSG:4326", width=width, height=height, format="GTiff", outputBounds=wgs84_raster_extent,
                    resampleAlg="nearest", outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"])
            
            # remove all of the permanent water data from outside of the label AOIs and save as raster
            with rasterio.open(f"{save_path}_wgs84.tif") as perm_water_file:
                perm_water_raster = perm_water_file.read(1)
                if scale_name != "local":
                    meta = perm_water_file.meta.copy()
                    meta.pop("nodata", None)
            if scale_name=="local":
                perm_water_raster = np.where(reference_label == 0, 0, perm_water_raster)
            with rasterio.open(f"{save_path}.tif", "w", **meta, compress="LZW") as file:
                file.write(perm_water_raster, 1)
                file.set_band_description(1, "permanent_water")
                file.nodata = None
                
            # delete the temporary intermediary files
            os.remove(f"{save_path}_utm.tif")
            os.remove(f"{save_path}_wgs84.tif")
            os.remove(f"{save_path}.geojson")
            os.remove(f"{save_path}.tif.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract data on permanent water from OSM.")

    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--download_global_permanent_water", action="store_true", default=False, help="Download global permanent water data from OSM")
    parser.add_argument("--create_permanent_water_rasters", action="store_true", default=False, help="Create permanent water rasters corresponding to the CEMS labels.")
    parser.add_argument("--scale", default="context", help="The scale at which to create raster files: local, context, or basin.")

    args = parser.parse_args()

    if args.download_global_permanent_water:
        download_global_permanent_water(args.data_folder, args.global_folder)

    if args.create_permanent_water_rasters:
        create_permanent_water_rasters(args.data_folder, args.global_folder, args.scale)