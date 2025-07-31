import geopandas as gpd
import os
from tqdm import tqdm
import rasterio
from rasterio.merge import merge
from osgeo import gdal
import numpy as np
import argparse
gdal.UseExceptions()

def create_lulc_rasters(data_folder, global_folder):
        
    # import metadata 
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    lulc_paths = gpd.read_file(f"{global_folder}/global_ESA_worldcover/geometries.geojson")
    lulc_folder = f"{data_folder}/full_subevent/raster_lulc"
    if not os.path.isdir(lulc_folder):
        os.mkdir(lulc_folder)

    for index in tqdm(range(len(raster_extents))):
        
        # read in the metadata for the subevent raster
        subevent = raster_extents["subevent"][index]
        geometry = raster_extents.loc[index, "geometry"]
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])

        lulc_path = f"{lulc_folder}/{subevent}.tif"
        if os.path.exists(lulc_path):
            continue

        # combine together the tiles that intersect the subevent raster
        paths = list(lulc_paths[lulc_paths.intersects(geometry)]["path"])
        tiles = [f"{global_folder}/global_ESA_worldcover/{path}" for path in paths]
        tiles_files = [rasterio.open(tile) for tile in tiles]
        merged_image, merged_transform = merge(tiles_files)
        merged_meta = tiles_files[0].meta.copy()
        merged_meta.update({"driver": "GTiff",
                            "height": merged_image.shape[1],
                            "width": merged_image.shape[2],
                            "transform": merged_transform})
        with rasterio.open(lulc_path, "w", **merged_meta) as file:
            file.write(merged_image)
        for file in tiles_files:
            file.close()

        # extract out the raster boundry extent from the LULC tiles
        gdal.Warp(lulc_path, lulc_path, format='GTiff', resampleAlg="nearest", width=width, height=height, outputBounds=geometry.bounds)

        # import the label raster to match the LULC raster to
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # remove data outside of the AOIs and save the final raster
        with rasterio.open(lulc_path) as read_file:
            lulc_raster = read_file.read(1)
            meta = read_file.meta.copy()
            meta.pop("nodata", None)
        lulc_raster = np.where(reference_label == 0, 0, lulc_raster)
        lulc_raster = lulc_raster/10
        with rasterio.open(lulc_path, "w", **meta, compress="LZW") as file:
            file.write(lulc_raster, 1)
            file.set_band_description(1, "LULC")
            file.nodata = None
        os.remove(f"{lulc_folder}/{subevent}.tif.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create LULC rasters for each subevent.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the global data folder.")
    args = parser.parse_args()

    create_lulc_rasters(args.data_folder, args.global_folder)