import geopandas as gpd
import os
from tqdm import tqdm
import argparse
import rasterio
from shapely.geometry import box

def main(raster_folder):
    
    extent_dict = {"event": [], "subevent":[], "date":[], "geometry":[]}

    # find the path to each raster, and create a dataframe containing its extent
    all_paths = [os.path.join(root, file) for root, dirs, files in os.walk(raster_folder) for file in files]
    for raster_path in tqdm(all_paths):
        extent_dict["event"].append(raster_path.split("/")[-1].split(".")[0].split("_")[0])
        extent_dict["subevent"].append(raster_path.split("/")[-1].split(".")[0])
        extent_dict["date"].append(raster_path.split("/")[-1].split(".")[0].split("_")[-1])
        extent_dict["geometry"].append(box(*rasterio.open(raster_path).bounds))

    extent = gpd.GeoDataFrame(extent_dict, crs="EPSG:4326")
    extent = extent.sort_values(["event", "subevent"], ascending=True, ignore_index=True)
    extent.to_file(f"metadata/raster_extent.geojson")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a GeoJSON file representing the extent of the rasters")
    parser.add_argument("--raster_folder", required=True, help="The path to the raster folder")
    args = parser.parse_args()

    main(raster_folder=args.raster_folder)