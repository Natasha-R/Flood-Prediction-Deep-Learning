import os
import geopandas as gpd
import pandas as pd
from osgeo import gdal
gdal.UseExceptions()
from tqdm import tqdm
import argparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def find_paths(event_folder, file):
    """Returns the paths in the event_folder for the given files. For example, folder=EMSR756, file=aoi.
    """
    search_patterns = {"aoi": [("areaOfInterestA", ".json"), ("area_of_interest", ".shp")],
                       "observed": [("observedEventA", ".json"), ("observed_event_a", ".shp"), ("crisis_information_poly", ".shp")],
                       "database": [("source", ".dbf")]}
    patterns = search_patterns[file]

    for file_name, format_type in patterns:
        paths = [os.path.join(root, file) for root, dirs, files in os.walk(event_folder) for file in files if file.endswith(format_type) and file_name in file]
        if paths:
            return paths
        
    raise Exception(f"No paths were found for {event_folder} {file}")

def find_subevent(match_file, paths):
    """Finds the path to the subevent contained in paths which matches the same subevent in match_file
       For example, if match_file is a path to an AOI, and paths is a list of dbf files,
       the dbf path to the subevent matching the subevent in the AOI path is returned.
    """
    subevent = "_".join(match_file.split("/")[-1].split("_")[:4])
    matching_paths = [path for path in paths if subevent in path]
    if len(matching_paths) != 1:
        raise Exception(f"Number of matching {paths[0].split('/')[-1].split('_')[4]} paths for observation {match_file}, is {len(matching_paths)}!")
    else:
        return matching_paths[0]
    
def create_raster_values(notation):
    """To be used with .apply on a dataframe. 
    Matches flooded area, trace, AOI and other to the associated value to be burned into the raster.
    """
    if notation == None:
        return 3
    elif notation.lower() == "flood trace" or notation.lower() == "flood traces":
        return 2
    elif notation.lower() == "flooded area" or notation.lower() == "dike breach" or notation == 2 or notation.lower() == "not applicable":
        return 3
    elif notation.lower() == "aoi":
        return 1
    else:
        return 0
    
def find_utm(lat_coord, lon_coord):
    """Given a latitude and longitude coordinate, find the corresponding UTM zone and return the coordinate system.
    """
    zone = min(int((lon_coord + 180) / 6) + 1, 60)
    if lat_coord >= 0:
        epsg = 32600 + zone  # WGS84 Northern Hemisphere
    else:
        epsg = 32700 + zone  # WGS84 Southern Hemisphere
    return epsg

def create_geojsons_by_date(cems_path, geojsons_folder):
    """
    Given a particular event, create geojsons containing the polygons for the AOI and observed events.
    A separate geojson file is created for each recording time.
    The files contain the geometry, attribute (flood trace, area, aoi, other), raster value, and date.
    """
    all_events = [event_code for event_code in os.listdir(cems_path) if "EMSR" in event_code]

    for event_code in tqdm(all_events):

        processed_obs = []
        event_path = os.path.join(cems_path, event_code)
        all_aoi = find_paths(event_path, "aoi")
        all_obs = find_paths(event_path, "observed")

        # older events have the source stored with the crisis polygon, instead of in a separate database
        older_event = True if "crisis" in all_obs[0] else False
        if not older_event:
            all_db = find_paths(event_path, "database")

        for obs_path in all_obs:

            # find the matching aoi, observed event, and database for the subevent
            aoi_path = find_subevent(obs_path, all_aoi)
            obs = gpd.read_file(obs_path).to_crs(epsg=4326)
            aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)

            # fix labelling errors in events
            if older_event:
                for gdf in [obs, aoi]:
                    gdf.rename(columns={"interpret":"notation", "subtype":"event_type"}, inplace=True)
            na_floods = (obs["event_type"].str.lower().str.contains("flood")) & (obs["notation"] == "Not Applicable")
            obs.loc[na_floods, "notation"] = "flooded area"

            # add dates of recording to the observed event
            if not older_event:
                db_path = find_subevent(obs_path, all_db)
                db = gpd.read_file(db_path)
                if "SRC_ID" in db.columns:
                    db.rename(columns={"SRC_ID": "src_id", "SRC_DATE":"src_date"}, inplace=True)
                obs = obs.merge(db, how="left", left_on="dmg_src_id", right_on="src_id")

            # add an aoi for the subevent for each recorded date
            aoi["notation"] = "AOI"
            all_dates = list(set(obs["src_date"]))
            aoi = pd.concat([aoi.head(1)]*len(all_dates), ignore_index=True)
            aoi["src_date"] = all_dates
            
            # create a single file for each subevent containing observed and aois
            aoi = aoi[["notation", "src_date", "geometry"]]
            obs = obs[["src_date", "notation", "geometry"]]
            obs = pd.concat([obs, aoi], ignore_index=True)
            processed_obs.append(obs)

        # create a single file for the whole event, with corresponding raster values
        processed_obs = pd.concat(processed_obs, ignore_index = True)
        processed_obs["raster_value"] = processed_obs.apply(lambda row: create_raster_values(row.notation), axis=1)

        # split the event by date recorded
        obs_by_date = processed_obs.groupby("src_date")
        for date, values in obs_by_date:
            
            # don't save events for which there is no recorded date
            missing_date_indicators = [None, "Not Applicable", "Not Applicable or dd/mm/aaaa", "0000/00/00", "1899/12/30", "30/12/1899", pd.Timestamp("1899-12-30 00:00:00")]
            if date in missing_date_indicators:
                continue

            # format the date string for the file name
            date = str(date).split(" ")[0]
            date = date.split("/")
            if len(date[-1])==4:
                date.reverse()
            date = "-".join(date)

            # project the coordinates to a system that uses metres, so the resolution can be calculated in metres
            coords = values.head(1).get_coordinates()
            coord_system = find_utm(coords.y.values[0], coords.x.values[0])
            proj_values = values.to_crs(epsg=coord_system)
            
            # save the subevent to a geojson file
            proj_values = proj_values.sort_values("raster_value", ascending=True)
            proj_values_path = f"{event_path.split('/')[-1]}_{date}"
            proj_values.to_file(f"{geojsons_folder}/{proj_values_path}.geojson", driver="GeoJSON")

def create_rasters_from_geojson(geojson_folder, raster_folder):
    """
    Convert the geojsons in the given folder into raster format.
    The rasters are all projected to EPSG:4326 (WGS84).
    """

    # find all geojson files in geojson_folder
    all_geojson = [os.path.join(root, file) for root, dirs, files in os.walk(geojson_folder) for file in files if file.endswith(".geojson")]

    for geojson_path in tqdm(all_geojson):

        subevent = geojson_path.split("/")[-1].split(".")[0]
        geojson = gpd.read_file(geojson_path)

        # rasterize the geojson, then project to EPSG:4326
        gdal.Rasterize(f"{raster_folder}/{subevent}_utm.tif", geojson_path, format="GTiff", xRes=10, yRes=10, attribute="raster_value")
        gdal.Warp(f"{raster_folder}/{subevent}.tif", f"{raster_folder}/{subevent}_utm.tif", srcSRS=geojson.crs, dstSRS="EPSG:4326", format="GTiff", outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"])
        os.remove(f"{raster_folder}/{subevent}_utm.tif")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate GeoJSON and raster files to represent the CEMS labels.")

    parser.add_argument('--cems_path', required=True, help="The path to the folder containing the original CEMS labels.")
    parser.add_argument("--geojson_folder", required=True, help="The path to the GeoJSON folder.")
    parser.add_argument("--raster_folder", default=None, help="The path to the raster folder.")

    parser.add_argument("--create_geojson", action="store_true", default=False, help="Create the GeoJSON files.")
    parser.add_argument("--create_raster", action="store_true", default=False, help="Create the rasters from the GeoJSON files.")

    args = parser.parse_args()

    if args.create_geojson:
        create_geojsons_by_date(args.cems_path, args.geojson_folder)

    if args.create_raster:
        if not args.raster_folder:
            raise Exception("Please specify a folder to save the rasters to.")
        create_rasters_from_geojson(args.geojson_folder, args.raster_folder)