import geopandas as gpd
import os
import pandas as pd
from tqdm import tqdm
import argparse
from shapely.ops import unary_union
import rasterio
import shapely
from shapely.geometry import box
import warnings
import sentinelhub
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS", category=UserWarning)

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
    for aoi_path in tqdm(aoi_paths, desc="Build metadata on each aoi"):
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
    for raster_path in tqdm(all_paths, desc="Build metadata on each raster"):
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
    subset = pd.DataFrame({"patch": patches, "event": events, "subevent": subevents})
    subset["subset"] = None
    subset.to_csv(f"{data_folder}/metadata/empty_data_subset.csv", index=False)

def determine_data_split(data_folder):
    """
    Create metadata describing the main split of the dataset into train, validation and test subsets.
    """
    # get all of the available patches
    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()

    # define the patches used for the validation and test subsets
    patch_subsets = {"val_patches": [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR465_2020-09-24", "EMSR517_2021-07-18", "EMSR634_2022-09-18"])][1::2],
                    "val_del_subevent": [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR788_2025-02-04"])],
                    "test_del_subevent": [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR763_2024-10-07"])],
                    "val_subevent": [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR770_2024-10-08"])],
                    "val_other": [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR764_2024-09-23"])],
                    "test_timing": [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR664_2023-05-21"])],
                    }
    eval_patches = [patch for patch_subset in patch_subsets.values() for patch in patch_subset]

    # exclude patches that are in subevents associated with the validation and test data, to prevent bias
    excluded = ["EMSR388_2019-09-18", # temporarily excluded while sentinel-2 data is unavailable
                "EMSR664_2023-05-17", "EMSR664_2023-05-18", "EMSR664_2023-05-20", "EMSR664_2023-05-22", # excluded from timing
                "EMSR465_2020-09-20", "EMSR517_2021-07-15", "EMSR517_2021-07-16", "EMSR517_2021-07-20", "EMSR517_2021-07-21" # excluded from patches
                "EMSR634_2022-09-16", "EMSR764_2024-09-30", # excluded from other
                "EMSR788_2025-02-05"] # excluded from del subevent due to bad annotation
    all_patches = [patch for patch in all_patches if not any(subevent in patch for subevent in excluded)]

    # include only events that are in Europe and which were not caused by snow melt
    # comment out to include all events within the training data
    events = pd.read_csv(f"{data_folder}/metadata/subevent_descriptions.csv")
    events = events[events["continent"]=="Europe"]
    events = events[events["flood_cause"] != "snow_melt"]
    included_events = set(events["event"])
    all_patches = [patch for patch in all_patches if any(subevent in patch for subevent in included_events)]

    # extract the training patches
    patch_subsets["train"] = list(set(all_patches)-set(eval_patches))

    # create the subset dataframe
    subset_df = pd.concat([pd.DataFrame({"patch":patches, "subset":[f"{patches_name}"]*len(patches)}) for patches_name, patches in patch_subsets.items()])
    subset_df["event"] = subset_df["patch"].str.split("_").str[0]
    subset_df["subevent"] = subset_df["patch"].str.split("_").str[0] + "_" + subset_df["patch"].str.split("_").str[1]
    subset_df = subset_df.to_csv(f"{data_folder}/subsets/europe_data_subset.csv", index=False)

    # create an alternate data subset for a single event only
    all_patches = os.listdir(f"{data_folder}/local/label")
    val_patches = [patch for patch in all_patches if any(subevent in patch for subevent in ["EMSR788_2025-02-04"])]
    train_patches = [patch for patch in all_patches if any(event in patch for event in ["EMSR788"])]
    train_patches = list(set(train_patches)-set(val_patches))
    subset_df = pd.concat([pd.DataFrame({"patch":val_patches, "subset":["val"]*len(val_patches)}),
                           pd.DataFrame({"patch":train_patches, "subset":["train"]*len(train_patches)})])
    subset_df["event"] = subset_df["patch"].str.split("_").str[0]
    subset_df["subevent"] = subset_df["patch"].str.split("_").str[0] + "_" + subset_df["patch"].str.split("_").str[1]
    subset_df.to_csv(f"{data_folder}/subsets/del_event_val_data_subset.csv", index=False)

def find_wider_scale_bounds(data_folder, global_folder):
    """
    Find the boundaries of the wider scales (context and basin) from the local patch bounds.
    The "context" scale corresponds to the level 12 basin, and the "basin" scale corresponds to the level 6 basin.
    """
    # import in the polygons for the basins at level 6 and 12
    print("Importing global river basin polygon reference data...")
    lvl6_basin = gpd.read_file(f"{global_folder}/global_basins/lev06_basin.geojson")
    lvl12_basin = gpd.read_file(f"{global_folder}/global_basins/lev12_basin.geojson")

    # import the metadata
    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()
    scales = {"patch":[], "basin_geometry":[], "context_geometry":[], "patch_geometry":[]}

    for patch in tqdm(all_patches, desc="Find context and basin scale boundaries"):

        # find the boundaries and the centre of each local 256x256 patch
        with rasterio.open(f"{data_folder}/local/label/" + patch) as file:
            patch_bounds = box(*(file.bounds))
            centre_x, centre_y = shapely.get_coordinates(patch_bounds.centroid)[0]

        for basin_name, basin in zip(["basin_geometry", "context_geometry"], [lvl6_basin, lvl12_basin]):

            # find the basin that intersects most with the patch
            intersecting_basins = basin[basin.geometry.intersects(patch_bounds)]
            intersecting_basins["geometry"] = shapely.make_valid(intersecting_basins["geometry"])
            intersecting_basins["intersection_area"] = intersecting_basins.geometry.intersection(patch_bounds).area
            intersecting_basins = intersecting_basins.sort_values("intersection_area", ascending=False)

            # if no basins intersect, then take the nearest basin instead
            if len(intersecting_basins) == 0:
                basin["distance"] = basin.geometry.distance(patch_bounds)
                intersecting_basins = basin.sort_values("distance", ascending=True).head(1)
                
            # determine a box that encapsulates the basin and has the patch in the centre
            minx, miny, maxx, maxy = intersecting_basins.reset_index(drop=True).bounds.iloc[0]
            box_half_size = max(max(abs(maxx - centre_x), abs(centre_x - minx)), max(abs(maxy - centre_y), abs(centre_y - miny)))
            basin_box = box(centre_x - box_half_size, centre_y - box_half_size, centre_x + box_half_size, centre_y + box_half_size)
            scales[basin_name].append(basin_box)

        scales["patch_geometry"].append(patch_bounds)
        scales["patch"].append(patch)

        # process the scale boundaries
        scales_gdf = gpd.GeoDataFrame(scales)
        scales_gdf["date"] = pd.to_datetime(scales_gdf["patch"].str.split("_").str[1])
        scales_gdf["height"] = 256
        scales_gdf["width"] = 256
        scales_gdf["subevent"] = scales_gdf["patch"].apply(lambda row: "_".join(row.split("_")[:2]))
        aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson").drop_duplicates(["event", "subevent", "event_date"]).reset_index(drop=True)[["subevent", "event_date"]]
        scales_gdf = scales_gdf.merge(aois, how="left", on="subevent").reset_index(drop=True)
        aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
        scales_gdf = gpd.GeoDataFrame(scales_gdf)
        for index in range(len(scales_gdf)):
            subevent_aois = aois[aois["subevent"]==scales_gdf.loc[index, "subevent"]]
            intersecting = subevent_aois[subevent_aois.geometry.intersects(scales_gdf.loc[index, "patch_geometry"])]
            if len(intersecting) == 0:
                subevent_aois["distance"] = subevent_aois.geometry.distance(scales_gdf.loc[index, "patch_geometry"])
                intersecting = subevent_aois.sort_values("distance", ascending=True)
            scales_gdf.loc[index, "geometry_event_date_id"] = intersecting.head(1)["geometry_event_date_id"].item()
            scales_gdf.loc[index, "context_resolution"] = min(sentinelhub.geo_utils.bbox_to_resolution(sentinelhub.BBox(scales_gdf.loc[index, "context_geometry"].bounds, crs="4326"), 256, 256, meters=True))
            scales_gdf.loc[index, "basin_resolution"] = min(sentinelhub.geo_utils.bbox_to_resolution(sentinelhub.BBox(scales_gdf.loc[index, "basin_geometry"].bounds, crs="4326"), 256, 256, meters=True))
            scales_gdf.loc[index, "geometry_event_date_id"] = intersecting.head(1)["geometry_event_date_id"].item()
        scales_gdf["geometry_event_date_id"] = scales_gdf["geometry_event_date_id"].astype("int")

        # save the boundaries as a geojson file
        scales_gdf.to_file(f"{data_folder}/metadata/scales.geojson")

    # find the area spanned by the context and basin geometries for each aoi
    ids, id_context_resolutions, id_basin_resolutions, context_union, basin_union = [], [], [], [], []
    for group_name, group_data in scales_gdf.groupby("geometry_event_date_id"):
        ids.append(group_name)
        id_context_resolutions.append(max(10, min(group_data["context_resolution"])))
        id_basin_resolutions.append(max(10, min(group_data["basin_resolution"])))
        context_union.append(unary_union(group_data["context_geometry"]))
        basin_union.append(unary_union(group_data["basin_geometry"]))
    id_gdf = gpd.GeoDataFrame({"geometry_event_date_id":ids, "context_resolution":id_context_resolutions, "basin_resolution":id_basin_resolutions, 
                    "context_geometry": context_union, "basin_geometry": basin_union})
    for geometry in ["context_geometry", "basin_geometry"]:
        multi = id_gdf[geometry].astype(str).str.contains("MULTI")
        id_gdf.loc[multi, geometry] = id_gdf.loc[multi, geometry].apply(lambda row : unary_union(row).convex_hull)
    id_gdf = id_gdf.merge(scales_gdf[["geometry_event_date_id", "date", "event_date", "subevent"]].drop_duplicates(), on="geometry_event_date_id", how="left")

    # save the aoi boundaries as a geojson file
    gpd.GeoDataFrame(id_gdf).to_file(f"{data_folder}/metadata/scales_aois.geojson")

    # create the folders for the context and basin patches
    if not os.path.isdir(f"{data_folder}/context"):
        os.mkdir(f"{data_folder}/context/")
    if not os.path.isdir(f"{data_folder}/basin"):
        os.mkdir(f"{data_folder}/basin/")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create GeoJSON files representing the extent of the CEMS AOIs and rasters")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--create_aoi_metadata", action="store_true", default=False, help="Create metadata describing the AOIs.")
    parser.add_argument("--create_raster_metadata", action="store_true", default=False, help="Create metadata describing the rasters.")
    parser.add_argument("--determine_data_split", action="store_true", default=False, help="Create metadata describing the split of the dataset into train, validation and test subsets.")
    parser.add_argument("--find_wider_scale_bounds", action="store_true", default=False, help="Find the boundaries of the wider scales from the local patch bounds.")
    
    args = parser.parse_args()

    if args.create_aoi_metadata:
        create_aoi_metadata(args.data_folder)

    if args.create_raster_metadata:
        create_raster_metadata(args.data_folder)

    if args.determine_data_split:
        determine_data_split(args.data_folder)

    if args.find_wider_scale_bounds:
        find_wider_scale_bounds(args.data_folder, args.global_folder)