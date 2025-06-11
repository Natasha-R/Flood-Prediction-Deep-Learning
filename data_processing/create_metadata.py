import geopandas as gpd
import os
import pandas as pd
from tqdm import tqdm
import argparse
from shapely.ops import unary_union
import rasterio
from shapely.geometry import box

def create_aoi_metadata(data_folder):
    """
    Create metadata corresponding to each individual AOI.
    This includes geographical extent boundaries and dates.
    Unique keys are also created for different combinations of geometries and dates.
    """

    cems_geojson_folder = f"{data_folder}/full_subevent/geojson_cems"
    extent_dict = {"event": [], "subevent":[], "date":[], "geometry":[]}

    # find the path to each aoi, and create a dataframe containing its extent
    aoi_paths = [os.path.join(root, file) for root, dirs, files in os.walk(cems_geojson_folder) for file in files]
    for aoi_path in tqdm(aoi_paths):
        aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
        aoi = aoi[aoi["raster_value"]==1]
        aoi["event"] = aoi_path.split("/")[-1].split(".")[0].split("_")[0]
        aoi["subevent"] = aoi_path.split("/")[-1].split(".")[0]
        aoi["date"] = aoi_path.split("/")[-1].split(".")[0].split("_")[-1]
        for attribute in ["event", "subevent", "date", "geometry"]:
            extent_dict[attribute] += list(aoi[attribute])
    extent = gpd.GeoDataFrame(extent_dict, crs="EPSG:4326")
    extent = extent.drop_duplicates(subset=["event", "subevent", "date", "geometry"], ignore_index=True)

    # merge any multipolygons into one
    multi = extent["geometry"].astype(str).str.contains("MULTI")
    extent.loc[multi, "geometry"] = extent.loc[multi, "geometry"].apply(unary_union)
    extent = extent.sort_values(["event", "subevent"], ascending=True, ignore_index=True)

    # for each flood event, find the earliest date that has recorded data (labels)
    events_dates = {"event": [], "event_date":[]}
    for event_name, event_values in extent.groupby("event"):
        events_dates["event"].append(event_name)
        events_dates["event_date"].append(min(list(event_values["date"])))
    events_dates = pd.DataFrame(events_dates)

    # calculate a 90 day window before the earliest data date
    events_dates["earlier_date"] = pd.to_datetime(events_dates["event_date"]) - pd.Timedelta(days=90)
    extent = extent.merge(events_dates, on="event")
    extent = extent.rename(columns={"date":"aoi_date"})

    # create a unique id for each aoi, based on the geometry and the (earliest) event date
    geometry_event_date_id = 0
    for group_name, group_values in extent.groupby(["geometry", "event_date"]):
        extent.loc[group_values.index, "geometry_event_date_id"] = int(geometry_event_date_id)
        geometry_event_date_id += 1
    extent["geometry_event_date_id"] = extent["geometry_event_date_id"].astype(int)

    # create a unique id for each aoi, based on the geometry
    geometry_id = 0
    for group_name, group_values in extent.groupby(["geometry"]):
        extent.loc[group_values.index, "geometry_id"] = int(geometry_id)
        geometry_id += 1
    extent["geometry_id"] = extent["geometry_id"].astype(int)

    extent = extent.sort_values(["event", "subevent"])
    metadata_folder = f"{data_folder}/metadata"
    if not os.path.isdir(metadata_folder):
        os.mkdir(metadata_folder)
    extent.to_file(f"{metadata_folder}/aoi_extent.geojson")

def create_raster_metadata(data_folder):
    """
    Create metadata corresponding to each raster.
    This includes geographical extent boundaries and dates.
    """
    
    cems_raster_folder = f"{data_folder}/raster_cems"
    extent_dict = {"event": [], "subevent":[], "date":[], "geometry":[], "height":[], "width":[]}

    # find the path to each raster, and create a dataframe containing its extent
    all_paths = [os.path.join(root, file) for root, dirs, files in os.walk(cems_raster_folder) for file in files]
    for raster_path in tqdm(all_paths):
        extent_dict["event"].append(raster_path.split("/")[-1].split(".")[0].split("_")[0])
        extent_dict["subevent"].append(raster_path.split("/")[-1].split(".")[0])
        extent_dict["date"].append(raster_path.split("/")[-1].split(".")[0].split("_")[-1])
        extent_dict["geometry"].append(box(*rasterio.open(raster_path).bounds))
        height, width = rasterio.open(raster_path).shape
        extent_dict["height"].append(height)
        extent_dict["width"].append(width)

    extent = gpd.GeoDataFrame(extent_dict, crs="EPSG:4326")
    extent = extent.sort_values(["event", "subevent"], ascending=True, ignore_index=True)
    metadata_folder = f"{data_folder}/metadata"
    extent.to_file(f"{metadata_folder}/raster_extent.geojson")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create GeoJSON files representing the extent of the CEMS AOIs and rasters")
    parser.add_argument("--data_folder", required=True, help="The path to the data folder.")
    parser.add_argument("--create_aoi_metadata", action="store_true", default=False, help="Create metadata describing the AOIs.")
    parser.add_argument("--create_raster_metadata", action="store_true", default=False, help="Create metadata describing the rasters.")
    args = parser.parse_args()

    if args.create_aoi_metadata:
        create_aoi_metadata(args.data_folder)

    if args.create_raster_metadata:
        create_raster_metadata(args.data_folder)