import geopandas as gpd
import os
import pandas as pd
from tqdm import tqdm
import rasterio
from shapely.geometry import box
import argparse
from shapely.ops import unary_union

def main(raster_extent, aoi_extent, geojson_folder=None, raster_folder=None):

    folder_path = raster_folder if raster_extent else geojson_folder
    file_name = "raster_extent" if raster_extent else "aoi_extent"
    all_paths = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files]
    extent_dict = {"event": [], "subevent":[], "date":[], "geometry":[]}

    if raster_extent:
        for raster_path in tqdm(all_paths):
            extent_dict["event"].append(raster_path.split("/")[-1].split(".")[0].split("_")[0])
            extent_dict["subevent"].append(raster_path.split("/")[-1].split(".")[0])
            extent_dict["date"].append(raster_path.split("/")[-1].split(".")[0].split("_")[-1])
            extent_dict["geometry"].append(box(*rasterio.open(raster_path).bounds))

    else: # if aoi_extent
        for aoi_path in tqdm(all_paths):
            aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
            aoi = aoi[aoi["raster_value"]==1]
            aoi["event"] = aoi_path.split("/")[-1].split(".")[0].split("_")[0]
            aoi["subevent"] = aoi_path.split("/")[-1].split(".")[0]
            aoi["date"] = aoi_path.split("/")[-1].split(".")[0].split("_")[-1]
            for attribute in ["event", "subevent", "date", "geometry"]:
                extent_dict[attribute] += list(aoi[attribute])

    extent = gpd.GeoDataFrame(extent_dict, crs="EPSG:4326")
    extent = extent.drop_duplicates(subset=["event", "subevent", "date", "geometry"], ignore_index=True)
    multi = extent["geometry"].astype(str).str.contains("MULTI")
    extent.loc[multi, "geometry"] = extent.loc[multi, "geometry"].apply(unary_union)
    extent = extent.sort_values(["event", "subevent"], ascending=True, ignore_index=True)

    if aoi_extent:
        events_dates = {"event": [], "event_date":[]}
        for event_name, event_values in extent.groupby("event"):
            events_dates["event"].append(event_name)
            events_dates["event_date"].append(min(list(event_values["date"])))
        events_dates = pd.DataFrame(events_dates)
        events_dates["earlier_date"] = pd.to_datetime(events_dates["event_date"]) - pd.Timedelta(days=90)
        extent = extent.merge(events_dates, on="event")
        extent = extent.rename(columns={"date":"aoi_date"})
        geometry_event_date_id = 0
        for group_name, group_values in extent.groupby(["geometry", "event_date"]):
            extent.loc[group_values.index, "geometry_event_date_id"] = int(geometry_event_date_id)
            geometry_event_date_id += 1
        extent["geometry_event_date_id"] = extent["geometry_event_date_id"].astype(int)

    extent = extent.sort_values(["event", "subevent"])
    extent.to_file(f"metadata/{file_name}.geojson")
    extent.to_csv(f"metadata/{file_name}.csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a GeoJSON file representing the extent of either each of the AOIs or rasters.")
    parser.add_argument("--geojson_folder", default=None, help="The path to the GeoJSON folder.")
    parser.add_argument("--raster_folder", default=None, help="The path to the raster folder.")
    parser.add_argument("--raster_extent", action="store_true", default=False, help="Create a GeoJSON file containing the extent of all of the rasters.")
    parser.add_argument("--aoi_extent", action="store_true", default=False, help="Create a GeoJSON file containing the extent of all of the aois.")
    args = parser.parse_args()

    if args.raster_extent:
        print("Finding the extent of all of the rasters:")
        main(raster_extent=True, aoi_extent=False, geojson_folder=args.geojson_folder, raster_folder=args.raster_folder)
    if args.aoi_extent:
        print("Finding the extent of all of the AOIs:")
        main(raster_extent=False, aoi_extent=True, geojson_folder=args.geojson_folder, raster_folder=args.raster_folder)