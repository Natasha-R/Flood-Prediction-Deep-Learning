import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal
import os
import rasterio
import numpy as np
import argparse
import shapely
gdal.UseExceptions()

def create_soil_vol_water(data_folder, global_folder, scale):
    """
    Create rasters representing the soil volumetric water content, corresponding to the CEMS label rasters.
    """

    # define the metadata and folder locations
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        soil_folder = f"{data_folder}/full_subevent/raster_soil_vol_water"
    else:
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        soil_folder = f"{data_folder}/{scale}/soil_vol_water"
    global_soil_10 = f"{global_folder}/global_vol_water/wv0010_15-30cm_mean.tif"
    global_soil_33 = f"{global_folder}/global_vol_water/wv0033_15-30cm_mean.tif"
    if not os.path.isdir(soil_folder):
        os.mkdir(soil_folder)

    for index in tqdm(range(len(raster_extents)), f"Create soil volumetric water content rasters for {scale} scale"):

        # import the metadata for the current raster
        bounds = raster_extents["geometry"][index].bounds
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])
        subevent = raster_extents["subevent"][index]
                
        # import the CEMS label as a reference to the extent of the AOIs
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)
        
        if scale == "local":
            soil_vol_water_raster_path = f"{soil_folder}/{subevent}"
            soil_vol_water_10_raster_path = f"{soil_folder}/{subevent}_10"
            soil_vol_water_33_raster_path = f"{soil_folder}/{subevent}_33"
        else:
            patch_name = raster_extents["patch"].iloc[index]
            soil_vol_water_raster_path = f"{soil_folder}/{patch_name}"
            soil_vol_water_10_raster_path = f"{soil_folder}/{patch_name}_10"
            soil_vol_water_33_raster_path = f"{soil_folder}/{patch_name}_33"
        
        if os.path.isfile(f"{soil_vol_water_raster_path}.tif"):
            continue

        # extract the given raster extents from the global rasters and impute the missing values using gdal_fillnodata
        for soil_vol_water_path, global_path in zip([soil_vol_water_10_raster_path, soil_vol_water_33_raster_path],
                                                    [global_soil_10, global_soil_33]):

            gdal.PushErrorHandler('CPLQuietErrorHandler')
            gdal.Warp(f"{soil_vol_water_path}_filled.tif", global_path, 
                    dstSRS="EPSG:4326", format='GTiff', resampleAlg="bilinear", outputBounds=bounds)
            gdal.PopErrorHandler()
            os.system(f"gdal_fillnodata -md=0 -q {soil_vol_water_path}_filled.tif {soil_vol_water_path}_filled.tif")
            gdal.Warp(f"{soil_vol_water_path}_filled.tif", f"{soil_vol_water_path}_filled.tif", 
                            srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                            resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            gdal.Warp(f"{soil_vol_water_path}.tif", global_path, 
                      dstSRS="EPSG:4326", format='GTiff', resampleAlg="bilinear", 
                      width=width, height=height, outputBounds=bounds)
            gdal.PopErrorHandler()
            
        # keep only data within the AOI bounds and replace no data with the filled version
        with rasterio.open(f"{soil_vol_water_10_raster_path}.tif") as soil_type_file:
            soil_vol_water_10 = soil_type_file.read(1)
            meta = soil_type_file.meta.copy()
            meta.pop("nodata", None)
            meta.update({"count": 2, "dtype": "uint16"})
        with rasterio.open(f"{soil_vol_water_10_raster_path}_filled.tif") as soil_type_file:
            soil_vol_water_10_filled = soil_type_file.read(1)
        soil_vol_water_10 = np.where(soil_vol_water_10 == -32768, soil_vol_water_10_filled, soil_vol_water_10)
        if scale == "local":
            soil_vol_water_10 = np.where(reference_label == 0, 0, soil_vol_water_10)

        with rasterio.open(f"{soil_vol_water_33_raster_path}.tif") as soil_type_file:
            soil_vol_water_33 = soil_type_file.read(1)
        with rasterio.open(f"{soil_vol_water_33_raster_path}_filled.tif") as soil_type_file:
            soil_vol_water_33_filled = soil_type_file.read(1)
        soil_vol_water_33 = np.where(soil_vol_water_33 == -32768, soil_vol_water_33_filled, soil_vol_water_33)
        if scale == "local":
            soil_vol_water_33 = np.where(reference_label == 0, 0, soil_vol_water_33)
        
        # save the final raster
        with rasterio.open(f"{soil_vol_water_raster_path}.tif", "w", **meta, compress="LZW") as file:
            file.write(soil_vol_water_10, 1)
            file.write(soil_vol_water_33, 2)
            file.set_band_description(1, "soil_vol_water_10")
            file.set_band_description(2, "soil_vol_water_33")
            file.nodata = None

        for path in [f"{soil_vol_water_10_raster_path}.tif.aux.xml", f"{soil_vol_water_33_raster_path}.tif.aux.xml", f"{soil_vol_water_raster_path}.tif.aux.xml",
                     f"{soil_vol_water_33_raster_path}.tif", f"{soil_vol_water_10_raster_path}.tif", 
                     f"{soil_vol_water_33_raster_path}_filled.tif", f"{soil_vol_water_10_raster_path}_filled.tif"]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create rasters for the soil volumetric water content.")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, nearby, context, con_context, or basin.")

    args = parser.parse_args()

    create_soil_vol_water(args.data_folder, args.global_folder, args.scale)
