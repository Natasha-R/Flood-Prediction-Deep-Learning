import geopandas as gpd
import os
from tqdm import tqdm
import pandas as pd
import tifffile as tf
import numpy as np
from scipy import stats
import argparse
from geopy.geocoders import Nominatim
import pycountry_convert as pc

def find_subevent_descriptions(data_folder, global_folder):
    """
    Create feature descriptions describing the characteristics of each subevent.
    """

    # import in necessary data and metadata
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson").to_crs("EPSG:6933")
    aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson").to_crs("EPSG:6933")
    events = pd.read_csv(f"{data_folder}/metadata/events.csv")
    lvl4_basin = gpd.read_file(f"{global_folder}/global_basins/lev04_basin.geojson").to_crs("EPSG:6933")
    lvl5_basin = gpd.read_file(f"{global_folder}/global_basins/lev05_basin.geojson").to_crs("EPSG:6933")
    geolocator = Nominatim(user_agent="geoapi")

    # define the characteristics used to describe each subevent
    characteristics = ["subevent", "mean_dem", "std_dem", "mean_precipitation_daily", "max_precipitation_daily", "std_precipitation_daily", "total_precipitation_daily",
                       "total_precipitation_15_28", "total_precipitation_29_42", "mean_precipitation_15_28", "mean_precipitation_29_42", "proportion_flood_label", 
                       "proportion_event_graded_label", "proportion_extent_versus_trace_label", "evaluation_label_quality", "proportion_perm_water",
                       "mean_soil_bulk_density", "mean_surface_soil_moisture_one_day", "mean_root_soil_moisture_one_day", "mean_surface_soil_moisture_one_week", 
                       "mean_root_soil_moisture_one_week", "mode_soil_class", "month", "year", "lvl4_basin_area", "lvl5_basin_area", "aoi_area", 
                       "proportion_urban", "continent", "country", "latitude", "longitude", "flood_cause"]
    subevent_descriptions = {characteristic : [] for characteristic in characteristics}

    for index in tqdm(range(len(raster_extents))):

        # read in the metadata for the subevent
        geometry = raster_extents.loc[index, "geometry"]
        event = raster_extents.loc[index, "event"]
        subevent = raster_extents.loc[index, "subevent"]
        subevent_descriptions["subevent"].append(subevent)
        reference = tf.imread(f"{data_folder}/full_subevent/raster_label/{subevent}.tif")

        ######## DEM

        dem = tf.imread(f"{data_folder}/full_subevent/raster_dem/{subevent}.tif")
        dem = dem[reference != 0]

        # mean DEM (general height)
        subevent_descriptions["mean_dem"].append(np.mean(dem).item())
        # standard deviation (mountainous/flat areas)
        subevent_descriptions["std_dem"].append(np.std(dem).item())

        ######## Soil

        # mean soil bulk density
        sbd = tf.imread(f"{data_folder}/full_subevent/raster_soil_bulk_density/{subevent}.tif")
        subevent_descriptions["mean_soil_bulk_density"].append(np.mean(sbd).item())

        # mean surface and root soil moisture (one day)
        sm_one_day = tf.imread(f"{data_folder}/full_subevent/raster_soil_moisture_one_day/{subevent}.tif")
        subevent_descriptions["mean_surface_soil_moisture_one_day"].append(np.mean(sm_one_day[:, :, 0][reference != 0]).item())
        subevent_descriptions["mean_root_soil_moisture_one_day"].append(np.mean(sm_one_day[:, :, 1][reference != 0]).item())

        # mean surface and root soil moisture (one week)
        sm_one_week = tf.imread(f"{data_folder}/full_subevent/raster_soil_moisture_one_week/{subevent}.tif")
        subevent_descriptions["mean_surface_soil_moisture_one_week"].append(np.mean(sm_one_week[:, :, 0][reference != 0]).item())
        subevent_descriptions["mean_root_soil_moisture_one_week"].append(np.mean(sm_one_week[:, :, 1][reference != 0]).item())

        # most common soil class
        soil_class = tf.imread(f"{data_folder}/full_subevent/raster_soil_class/{subevent}.tif")
        subevent_descriptions["mode_soil_class"].append(stats.mode(soil_class[reference != 0])[0].item())

        ######## Precipitation

        precipitation = tf.imread(f"{data_folder}/full_subevent/raster_precipitation/{subevent}.tif")
        precipitation_daily = precipitation[:, :, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)][reference != 0]
        precipitation_15_28 = precipitation[:, :, 14][reference != 0]
        precipitation_29_42 = precipitation[:, :, 15][reference != 0]

        # mean daily precipitation
        subevent_descriptions["mean_precipitation_daily"].append(np.mean(precipitation_daily).item())
        # max daily precipitation
        subevent_descriptions["max_precipitation_daily"].append(np.max(precipitation_daily).item())
        # std of daily precipitation
        subevent_descriptions["std_precipitation_daily"].append(np.std(precipitation_daily).item())
        # total rainfall over the daily precipitation
        subevent_descriptions["total_precipitation_daily"].append(np.sum(precipitation_daily).item())
        # total rainfall over days 15-28
        subevent_descriptions["total_precipitation_15_28"].append(np.sum(precipitation_15_28).item())
        # mean rainfall over days 15-28
        subevent_descriptions["mean_precipitation_15_28"].append(np.mean(precipitation_15_28).item())
        # total rainfall over days 29-42
        subevent_descriptions["total_precipitation_29_42"].append(np.sum(precipitation_29_42).item())
        # mean rainfall over days 29-42
        subevent_descriptions["mean_precipitation_29_42"].append(np.mean(precipitation_29_42).item())

        ######## Label

        num_trace = np.sum(reference == 2).item()
        num_extent = np.sum(reference == 3).item()
        num_flood = num_trace + num_extent
        num_aoi = np.sum(reference == 1).item()
        num_total = num_flood+num_aoi

        # proportion of flooding in the label
        subevent_descriptions["proportion_flood_label"].append(num_flood / num_total)
        if num_flood == 0: 
            subevent_descriptions["proportion_extent_versus_trace_label"] = 0
        else:
            subevent_descriptions["proportion_extent_versus_trace_label"].append(num_extent/num_flood)

        # proportion of graded labels in the full event (quality of labels)
        all_aois = os.listdir(f"{data_folder}/raw_cems/{event}")
        all_del = [aoi for aoi in all_aois if "_DEL_" in aoi or "DELINEATION" in aoi]
        num_aois = len(all_aois)
        num_del = len(all_del)
        num_grad = num_aois - num_del
        subevent_descriptions["proportion_event_graded_label"].append(num_grad/num_aois)

        # evaluation of label quality 
        subevent_descriptions["evaluation_label_quality"].append(events[events["Code"]==event]["Label"].item())

        ######## Permanent water

        perm_water = tf.imread(f"{data_folder}/full_subevent/raster_permanent_water/{subevent}.tif")
        perm_water = perm_water[reference != 0]
        num_perm_water = np.sum(reference == 1).item()
        num_non_water = np.sum(reference == 0).item()

        # proportion of permanent water in the label
        subevent_descriptions["proportion_perm_water"].append(num_perm_water / (num_perm_water + num_non_water))

        ######## Land use

        lulc = tf.imread(f"{data_folder}/full_subevent/raster_lulc/{subevent}.tif")
        lulc = lulc[reference != 0]

        # proportion of urban/built-up areas in the subevent area
        num_urban = np.sum(lulc == 5).item()
        num_not_urban = np.sum(lulc != 5).item()
        subevent_descriptions["proportion_urban"].append(num_urban / (num_urban + num_not_urban))

        ######## Geographical area

        centre = raster_extents.to_crs("EPSG:4326").loc[index, "geometry"].centroid
        longitude = centre.x
        latitude = centre.y
        subevent_descriptions["longitude"].append(longitude)
        subevent_descriptions["latitude"].append(latitude)
        
        # country and continent
        country_code = geolocator.reverse((latitude, longitude)).raw["address"]["country_code"].upper()
        country_name = pc.country_alpha2_to_country_name(country_code)
        continent_name = pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(country_code))
        subevent_descriptions["country"].append(country_name)
        subevent_descriptions["continent"].append(continent_name)

        # date
        subevent_descriptions["month"].append(raster_extents.loc[0, "date"].month)
        subevent_descriptions["year"].append(raster_extents.loc[0, "date"].year)

        # level 4 and 5 basin areas
        for basin, basin_name in zip([lvl4_basin, lvl5_basin], ["lvl4_basin_area", "lvl5_basin_area"]):
            intersecting = basin[basin.geometry.intersects(geometry)]
            intersecting["intersection_area"] = intersecting.geometry.intersection(geometry).area
            basin_area = intersecting.sort_values("intersection_area", ascending=False).reset_index(drop=True).head(1)["SUB_AREA"].item()
            subevent_descriptions[basin_name].append(basin_area)

        # geographical size of the AOI areas in the raster
        subevent_aois = aois[aois["subevent"]==subevent]
        subevent_descriptions["aoi_area"].append(np.sum(subevent_aois["geometry"].area).item())

        # cause of the flood
        subevent_descriptions["flood_cause"].append(events[events["Code"]==event]["Cause"].item())

    # save all of the subevent descriptions to a csv file
    pd.DataFrame(subevent_descriptions).to_csv(f"{data_folder}/metadata/subevent_descriptions.csv")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create feature descriptions describing the characteristics of each subevent.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the global data folder.")
    args = parser.parse_args()

    find_subevent_descriptions(args.data_folder, args.global_folder)