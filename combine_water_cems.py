import argparse
import geopandas as gpd
from tqdm import tqdm
import rasterio
import numpy as np

def combine_permanent_water_and_labels(perm_water_raster_folder, cems_raster_folder, label_raster_folder):

    raster_extents = gpd.read_file("metadata/raster_extent.geojson")

    for index in tqdm(range(len(raster_extents))):
        
        # load in the rasters
        subevent = raster_extents["subevent"][index]
        with rasterio.open(f"{cems_raster_folder}/{subevent}.tif") as cems_file:
            cems_raster = cems_file.read(1)
            meta = cems_file.meta.copy()
        with rasterio.open(f"{perm_water_raster_folder}/{subevent}.tif") as perm_water_file:
            perm_water_raster = perm_water_file.read(1)

        # remove all flood labels from locations of permanent water and save as a raster
        label_raster = np.where(perm_water_raster == 1, 1, cems_raster)
        with rasterio.open(f"{label_raster_folder}/{subevent}.tif", "w", **meta, compress="LZW") as file:
            file.write(label_raster, 1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine the permanent water and CEMS data to create the labels.")

    parser.add_argument("--perm_water_raster_folder", required=True, help="The path to the permanent water raster folder.")
    parser.add_argument("--cems_raster_folder", required=True, help="The path to the CEMS raster folder.")
    parser.add_argument("--label_raster_folder", required=True, help="The path to the folder in which to save the created labels.")

    args = parser.parse_args()

    args.perm_water_raster_folder = "/mnt/datadisk/natasha_flood/data/raster_perm_water/"
    args.cems_raster_folder = "/mnt/datadisk/natasha_flood/data/raster_cems/"
    args.label_raster_folder = "/mnt/datadisk/natasha_flood/data/raster_label/"

    combine_permanent_water_and_labels(args.perm_water_raster_folder, args.cems_raster_folder, args.label_raster_folder)
    