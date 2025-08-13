from data_processing.labels import create_cems_geojson, create_cems_raster, combine_cems_and_permanent_water
from data_processing.metadata import create_aoi_metadata, create_raster_metadata
from data_processing.permanent_water import download_global_permanent_water, create_permanent_water_rasters
from data_processing.dem import extract_dem_aoi, create_dem_rasters
from data_processing.sentinel2 import find_sentinel2_availability, find_minimal_cloud_cover, download_sentinel2, create_sentinel2_rasters
from data_processing.soil_moisture import download_soil_moisture_data, create_soil_moisture_rasters
from data_processing.soil_type import create_soil_type
from data_processing.sentinel1 import find_sentinel1_availability, download_sentinel1, create_sentinel1_aoi_date_difference, create_sentinel1_rasters
from data_processing.precipitation import download_precipitation, create_precipitation_rasters
from data_processing.local_patches import create_label_local_patches, create_features_local_patches, find_wider_scale_bounds
import os
import argparse

##### PRELIMINARY SET-UP
# 1. Put the raw downloaded CEMS data into a folder called "raw_cems" in "data_folder"
# 2. Download the FABDEM data from https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn and extract all the image files. Place in "global_folder"/global_fabdem.
# 3. Download the seas and waters polygons from https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip and extract the shapefiles. Place in "global_folder"/global_seas_polygons.
# 4. Download the global soil classes and global soil bulk density data from https://soilgrids.org/. Place the files in "global_folder".
# 5. Download global ESA worldcover.
# 6. Download global basins.

def main(data_folder, global_folder):

    # Process the CEMS data
    print("Processing the labels...")
    create_cems_geojson(data_folder)
    create_cems_raster(data_folder)

    # Create metadata describing the AOIs and rasters of the CEMS data
    create_aoi_metadata(data_folder)
    create_raster_metadata(data_folder)

    # Create local 256x256 patches of the labels and 
    # find the bounds of the wider context and basin scales
    print("Creating local patches from the labels...")
    create_label_local_patches(data_folder)
    find_wider_scale_bounds(data_folder, global_folder)

    # Create the permanent water rasters and modify the CEMS rasters to create the final labels
    # download_global_permanent_water(data_folder, global_folder)
    create_permanent_water_rasters(data_folder, global_folder)
    combine_cems_and_permanent_water(data_folder)
    
    # Create the DEM rasters
    print("Creating DEM rasters...")
    extract_dem_aoi(data_folder, global_folder)
    create_dem_rasters(data_folder)

    # Create the soil moisture and soil type rasters
    print("Creating soil moisture and soil type rasters...")
    download_soil_moisture_data(data_folder, global_folder)
    create_soil_moisture_rasters(data_folder, global_folder)
    create_soil_type(data_folder, global_folder)

    # Create the precipitation rasters
    print("Creating precipitation rasters...")
    download_precipitation(data_folder, global_folder)
    create_precipitation_rasters(data_folder, global_folder)

    # Create the Sentinel 2 rasters
    print("Downloading and processing Sentinel 2 data...")
    find_sentinel2_availability(data_folder)
    find_minimal_cloud_cover(data_folder)
    download_sentinel2(data_folder)
    create_sentinel2_rasters(data_folder)

    # Create the Sentinel 1 rasters
    print("Downloading and processing Sentinel 1 data...")
    find_sentinel1_availability(data_folder)
    download_sentinel1(data_folder) 
    create_sentinel1_aoi_date_difference(data_folder) 
    create_sentinel1_rasters(data_folder)

    # Create local 256x256 patches of the features
    print("Creating local patches of the features...")
    create_features_local_patches(data_folder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create the full data stack for all CEMS data stored in 'data_folder'.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    
    args = parser.parse_args()

    if not os.path.isdir(f"{args.data_folder}/raw_cems"):
        raise FileNotFoundError("CEMS labels must be put in the raw_cems folder in", args.data_folder)
    if len(os.listdir(f"{args.data_folder}/raw_cems"))==0:
        raise FileNotFoundError("raw_cems folder is empty")

    main(data_folder=args.data_folder, global_folder=args.global_folder)