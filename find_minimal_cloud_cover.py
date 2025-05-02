import pandas as pd
pd.set_option('mode.chained_assignment', None)
import geopandas as gpd
from tqdm import tqdm

# import the aoi extent and the sentinel 2 availability metadata
aois = gpd.read_file("metadata/aoi_extent.geojson")
availability = pd.read_csv("metadata/sentinel2_availability.csv")

# find the optimal date for Sentinel 2 data for each aoi, based on the cloud cover percentages
cloud_cover_availability = {"geometry_event_date_id":[], "availability_date":[], "cloud_cover_percentage":[]}
for geometry_event_date_id, data in tqdm(availability.groupby("geometry_event_date_id")):

    # only use dates for which all tiles within the aoi are available on that date
    dates_on_which_all_tiles_available = list(data["tile_date"].value_counts()[data["tile_date"].value_counts()==data["tile"].nunique()].index)
    if len(dates_on_which_all_tiles_available) == 0:
        print("No dates on which all tiles are available for event", geometry_event_date_id)
        continue
    data = data[data["tile_date"].isin(dates_on_which_all_tiles_available)]

    event_date = pd.to_datetime(data["event_date"].head(1).item())
    data["tile_date"] = pd.to_datetime(data["tile_date"])

    # find the average cloud cover across all of the tiles, per date
    cloud_cover_means = data.groupby(["tile_date"])["cloud_cover"].mean().reset_index()

    # the optimum date, based on the amount of cloud cover, fits the following criteria:
    # if cloud cover is under 5% within 30 days of the flood event, or if cloud cover is under 10% within 30 days of the flood event
    # if this criteria isn't met, then compelte the same search within 60 days, and then 90 days
    # if this criteria isn't met, then simply return the date with the lowest cloud cover percentage.
    selected_cloud_cover = pd.DataFrame()
    for window in [30, 60, 90]:
        cloud_cover_means_within_window = cloud_cover_means[cloud_cover_means["tile_date"] > event_date - pd.Timedelta(days=window)]
        for threshold in [5, 10]:
            result = cloud_cover_means_within_window[cloud_cover_means_within_window["cloud_cover"] < threshold].sort_values("tile_date", ascending=False)
            if not result.empty:
                selected_cloud_cover = result
                break
        if not selected_cloud_cover.empty:
            break
    if selected_cloud_cover.empty:
        selected_cloud_cover = cloud_cover_means.sort_values("cloud_cover", ascending=True)

    cloud_cover_availability["geometry_event_date_id"].append(geometry_event_date_id)
    cloud_cover_availability["availability_date"].append(selected_cloud_cover.head(1)["tile_date"].item())
    cloud_cover_availability["cloud_cover_percentage"].append(selected_cloud_cover.head(1)["cloud_cover"].item())

cloud_cover_availability = pd.DataFrame(cloud_cover_availability)
aois = aois.merge(cloud_cover_availability, on="geometry_event_date_id")

aois.to_file(f"metadata/aoi_availability.geojson")