import geopandas as gpd
import os
from tqdm import tqdm
import rasterio
from osgeo import gdal
import numpy as np
import shapely
import argparse
gdal.UseExceptions()

def create_hand_rasters(data_folder, global_folder, scale):
    """
    Create HAND (height above nearest drainage) rasters.
    """
        
    # import metadata 
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        hand_folder = f"{data_folder}/full_subevent/raster_hand"
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        hand_folder = f"{data_folder}/{scale}/hand"
    hand_boundaries = gpd.read_file(f"{global_folder}/global_hand/hand_boundaries.geojson")

    if not os.path.isdir(hand_folder):
        os.mkdir(hand_folder)

    for index in tqdm(range(len(raster_extents)), f"Create HAND rasters for {scale} scale"):
        
        # read in the metadata for the subevent raster
        subevent = raster_extents["subevent"][index]
        geometry = raster_extents.loc[index, "geometry"]
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])

        if scale == "local":
            hand_path = f"{hand_folder}/{subevent}.tif"
        else: # if scale == "context" or scale == "basin"
            patch = raster_extents["patch"].iloc[index]
            hand_path = f"{hand_folder}/{patch}"
        if os.path.exists(hand_path):
            continue

        # combine together the tiles that intersect the raster
        tiles = list(hand_boundaries[hand_boundaries.intersects(geometry)]["tile"])
        tiles = [f"{global_folder}/global_hand/{path}" for path in tiles]
        gdal.Warp(hand_path, tiles, srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', 
                  outputType=gdal.GDT_UInt16, creationOptions=["COMPRESS=LZW"],
                  resampleAlg="bilinear", width=width, height=height, outputBounds=geometry.bounds)

        # import the label raster to match the HAND raster to
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # remove data outside of the AOIs and save the final raster
        with rasterio.open(hand_path) as read_file:
            hand_raster = read_file.read(1)
            meta = read_file.meta.copy()
            meta.pop("nodata", None)
        if scale == "local":
            hand_raster = np.where(reference_label == 0, 0, hand_raster)

        with rasterio.open(hand_path, "w", **meta, compress="LZW") as file:
            file.write(hand_raster, 1)
            file.set_band_description(1, "HAND")
            file.nodata = None
        os.remove(f"{hand_path}.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create HAND rasters.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the global data folder.")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, context, or basin.")
    args = parser.parse_args()

    create_hand_rasters(args.data_folder, args.global_folder, args.scale)