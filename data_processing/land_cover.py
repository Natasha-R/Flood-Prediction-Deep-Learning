import geopandas as gpd
import os
from tqdm import tqdm
import rasterio
from rasterio.merge import merge
from osgeo import gdal
import numpy as np
import shapely
import argparse
gdal.UseExceptions()

def create_land_cover_rasters(data_folder, global_folder, scale):
    """
    Create land_cover rasters.
    """
        
    # import metadata 
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        land_cover_folder = f"{data_folder}/full_subevent/raster_land_cover"
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        land_cover_folder = f"{data_folder}/{scale}/land_cover"
    land_cover_paths = gpd.read_file(f"{global_folder}/global_ESA_worldcover/geometries.geojson")
    if not os.path.isdir(land_cover_folder):
        os.mkdir(land_cover_folder)

    for index in tqdm(range(len(raster_extents))):
        
        # read in the metadata for the subevent raster
        subevent = raster_extents["subevent"][index]
        geometry = raster_extents.loc[index, "geometry"]
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])

        if scale == "local":
            land_cover_path = f"{land_cover_folder}/{subevent}.tif"
        else: # if scale == "context" or scale == "basin"
            patch = raster_extents["patch"].iloc[index]
            land_cover_path = f"{land_cover_folder}/{patch}"
        if os.path.exists(land_cover_path):
            continue

        # combine together the tiles that intersect the raster
        paths = list(land_cover_paths[land_cover_paths.intersects(geometry)]["path"])
        tiles = [f"{global_folder}/global_ESA_worldcover/{path}" for path in paths]
        gdal.Warp(land_cover_path, tiles, srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', 
                  outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"],
                  resampleAlg="nearest", width=width, height=height, outputBounds=geometry.bounds)

        # import the label raster to match the land cover raster to
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # remove data outside of the AOIs and save the final raster
        with rasterio.open(land_cover_path) as read_file:
            land_cover_raster = read_file.read(1)
            meta = read_file.meta.copy()
            meta.pop("nodata", None)
        if scale == "local":
            land_cover_raster = np.where(reference_label == 0, 0, land_cover_raster)
        land_cover_raster = land_cover_raster/10
        with rasterio.open(land_cover_path, "w", **meta, compress="LZW") as file:
            file.write(land_cover_raster, 1)
            file.set_band_description(1, "land_cover")
            file.nodata = None
        os.remove(f"{land_cover_path}.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create land_cover rasters.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the global data folder.")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, context, or basin.")
    args = parser.parse_args()

    create_land_cover_rasters(args.data_folder, args.global_folder, args.scale)