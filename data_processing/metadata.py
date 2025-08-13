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
    
    cems_raster_folder = f"{data_folder}/full_subevent/raster_cems"
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

def create_empty_patch_csv(data_folder):
    """
    Create an empty csv, containing a row for each patch.
    """
    patches = os.listdir(f"{data_folder}/local/label")
    patches.sort()
    events = [patch.split("_")[0] for patch in patches]
    subevents = ["_".join(patch.split("_")[:2]) for patch in patches]
    subset = pd.DataFrame({"patch": patches, "event": events, "subevents": subevents})
    subset["subset"] = None
    subset.to_csv(f"{data_folder}/metadata/empty_data_subset.csv", index=False)

def determine_data_split(data_folder):
    """
    Create metadata describing the main split of the dataset into train, validation and test subsets.
    """
    # read in the metadata
    desc = pd.read_csv(f"{data_folder}/metadata/subevent_descriptions.csv")
    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()

    # determine the patches that will form the 'subevent' validation and test sets
    subevent_test = ["EMSR561_2022-01-29", "EMSR561_2022-02-01", "EMSR631_2022-09-11", "EMSR631_2022-09-23", 
                    "EMSR692_2023-09-07", "EMSR692_2023-09-10", "EMSR692_2023-09-12", "EMSR692_2023-09-22",
                    "EMSR720_2024-05-05", "EMSR720_2024-05-07", "EMSR720_2024-05-15", "EMSR773_2024-10-31", 
                    "EMSR773_2024-11-03", "EMSR773_2024-11-10", "EMSR773_2024-11-18", "EMSR774_2024-10-30", 
                    "EMSR774_2024-11-03", "EMSR774_2024-11-05", "EMSR774_2024-11-08"]
    subevent_val = ["EMSR692_2023-09-15", "EMSR692_2023-09-18", "EMSR773_2024-11-06", "EMSR773_2024-11-15"]
    subevent_test_patches = [patch for patch in all_patches if any(subevent in patch for subevent in subevent_test)]
    subevent_val_patches = [patch for patch in all_patches if any(subevent in patch for subevent in subevent_val)]

    # determine the patches that will form the 'timing' test set
    timing_test = ["EMSR555", "EMSR762", "EMSR637", "EMSR429", "EMSR768"]
    timing_test_patches = [patch for patch in all_patches if any(subevent in patch for subevent in timing_test)]

    # determine the patches that will form the 'other' test and validation sets
    other_test = ["EMSR754", "EMSR779", "EMSR788", "EMSR570", "EMSR763", "EMSR764", "EMSR759", 
                "EMSR756", "EMSR706", "EMSR758", "EMSR722", "EMSR431", "EMSR650"]
    other_val = ["EMSR487", "EMSR663", "EMSR694"]
    other_test_patches = [patch for patch in all_patches if any(subevent in patch for subevent in other_test)]
    other_val_patches = [patch for patch in all_patches if any(subevent in patch for subevent in other_val)]

    # determine the patches that will form the 'patch' validation and test sets
    splitting = list(desc[(desc["event"].isin(["EMSR774", "EMSR773", "EMSR720", "EMSR561", "EMSR631", "EMSR692"])) & ~desc["event"].isin(subevent_test) & ~desc["event"].isin(subevent_val)]["subevent"])
    patches_eval_patches = [patch for subevent in splitting for index, patch in enumerate([patch for patch in all_patches if subevent in patch]) if index % 2 == 0]
    patches_val_patches = [patch for index, patch in enumerate([patch for patch in patches_eval_patches]) if index % 2 == 0]
    patches_test_patches = [patch for index, patch in enumerate([patch for patch in patches_eval_patches]) if index % 2 != 0]

    # determine the patches that will form the training set
    eval_patches = subevent_test_patches + subevent_val_patches + timing_test_patches + other_test_patches + other_val_patches + patches_val_patches + patches_test_patches
    training_patches = list(set(all_patches)-set(eval_patches))

    # create_empty_patch_csv(data_folder)

    # save the assignment of the patches to the data subsets in a csv file
    subset = pd.read_csv(f"{data_folder}/metadata/empty_data_subset.csv")
    subset.loc[subset["patch"].isin(training_patches), "subset"] = "train"
    subset.loc[subset["patch"].isin(other_test_patches), "subset"] = "test_other"
    subset.loc[subset["patch"].isin(subevent_test_patches), "subset"] = "test_subevent"
    subset.loc[subset["patch"].isin(patches_test_patches), "subset"] = "test_patches"
    subset.loc[subset["patch"].isin(timing_test_patches), "subset"] = "test_timing"
    subset.loc[subset["patch"].isin(other_val_patches), "subset"] = "val_other"
    subset.loc[subset["patch"].isin(subevent_val_patches), "subset"] = "val_subevent"
    subset.loc[subset["patch"].isin(patches_val_patches), "subset"] = "val_patches"
    subset.to_csv(f"{data_folder}/metadata/data_subset.csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create GeoJSON files representing the extent of the CEMS AOIs and rasters")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--create_aoi_metadata", action="store_true", default=False, help="Create metadata describing the AOIs.")
    parser.add_argument("--create_raster_metadata", action="store_true", default=False, help="Create metadata describing the rasters.")
    parser.add_argument("--determine_data_split", action="store_true", default=False, help="Create metadata describing the split of the dataset into train, validation and test subsets.")
    
    args = parser.parse_args()

    if args.create_aoi_metadata:
        create_aoi_metadata(args.data_folder)

    if args.create_raster_metadata:
        create_raster_metadata(args.data_folder)

    if args.determine_data_split:
        determine_data_split(args.data_folder)