from data_processing.process_cems import create_cems_geojson, create_cems_raster
from data_processing.create_metadata import find_aoi_extents, find_raster_extents
from data_processing.modify_cems_permanent_water import create_permanent_water_geojson, create_permanent_water_raster, combine_cems_and_permanent_water
from data_processing.create_dem import create_dem_rasters
from data_processing.create_sentinel2 import find_sentinel2_availability, find_minimal_cloud_cover, download_sentinel2
from data_processing.create_soil_moisture import create_soil_moisture_rasters

import argparse

##### PRELIMINARY SET-UP
# 1. Put the raw downloaded CEMS data into a folder called "raw_cems" in "data_folder"
# 2. Download the FABDEM data from https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn and extract all the image files.
# 3. Download the seas and waters polygons from https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip, and place the unzipped folder into "water-polygons-split-4326" in "data_folder".

def main(data_folder, fabdem_folder):
    
    # Process the CEMS data
    create_cems_geojson(data_folder)
    create_cems_raster(data_folder)

    # Create metadata describing the AOIs and rasters of the CEMS data
    find_aoi_extents(data_folder)
    find_raster_extents(data_folder)

    # Modify the CEMS rasters with permanent water indicators, to create the final labels
    create_permanent_water_geojson(data_folder)
    create_permanent_water_raster(data_folder)
    combine_cems_and_permanent_water(data_folder)

    # Create the DEM rasters
    create_dem_rasters(fabdem_folder, data_folder)

    # Create the Sentinel 2 rasters
    find_sentinel2_availability(data_folder)
    find_minimal_cloud_cover(data_folder)
    download_sentinel2(data_folder)

    # Create the soil moisture rasters
    create_soil_moisture_rasters(data_folder)

    # Create the precipitation rasters

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create the full data stack for all CEMS data stored in 'data_folder'.")
    parser.add_argument("--fabdem_folder", default=None, help="The path to the folder containing the full downloaded FABDEM.")
    parser.add_argument("--data_folder", required=True, help="The path to the data folder.")
    
    args = parser.parse_args()

    main(data_folder=args.data_folder, fabdem_folder=args.fabdem_folder)