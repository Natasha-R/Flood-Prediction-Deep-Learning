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

def find_wider_scale_bounds(data_folder, global_folder):
    """
    Find the boundaries of the wider scales (context and basin) from the local patch bounds.
    The "context" scale corresponds to the level 12 basin, and the "basin" scale corresponds to the level 6 basin.
    """
    # import in the polygons for the basins at level 6 and 12
    lvl6_basin = gpd.read_file(f"{global_folder}/global_basins/lev06_basin.geojson")
    lvl12_basin = gpd.read_file(f"{global_folder}/global_basins/lev12_basin.geojson")

    # import the metadata
    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()
    scales = {"patch":[], "basin_geometry":[], "context_geometry":[], "patch_geometry":[]}

    for patch in tqdm(all_patches):

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
        for index in tqdm(range(len(scales_gdf))):
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
    id_gdf = id_gdf.merge(scales_gdf[["geometry_event_date_id", "date"]].drop_duplicates(), on="geometry_event_date_id", how="left")

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