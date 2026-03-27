import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal
import os
import rasterio
import numpy as np
import argparse
import shapely
gdal.UseExceptions()

def create_flow_direction(data_folder, global_folder, scale):
    """
    Create rasters representing the flow direction data, corresponding to the CEMS label rasters.
    """

    # define the metadata and folder locations
    global_flow_direction = f"{global_folder}/global_flow_direction/global_flow_direction_3s.tif"
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        flow_dir_folder = f"{data_folder}/full_subevent/raster_flow_direction"
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        flow_dir_folder = f"{data_folder}/{scale}/flow_direction"
        if scale == "basin":
            global_flow_direction = f"{global_folder}/global_flow_direction/global_flow_direction_30s.tif"
    if not os.path.isdir(flow_dir_folder):
        os.mkdir(flow_dir_folder)

    for index in tqdm(range(len(raster_extents)), desc=f"Create flow direction raster for {scale} scale"):

        # import the metadata for the current raster
        bounds = raster_extents["geometry"][index].bounds
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])
        subevent = raster_extents["subevent"][index]

        if scale == "local":
            save_path = f"{flow_dir_folder}/{subevent}.tif"
        else:  # if scale == "context" or scale == "basin"
            patch_name = raster_extents["patch"].iloc[index]
            save_path = f"{flow_dir_folder}/{patch_name}"
        
        if os.path.exists(save_path):
            continue

        # import the CEMS label as a reference to the extent of the AOIs
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # extract the extent bounds from the global flow direction raster
        gdal.Warp(save_path, global_flow_direction, srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                    resampleAlg="nearest", width=width, height=height, outputBounds=bounds)
        
        with rasterio.open(save_path) as file:
            fd_raster = file.read(1)
            meta = file.meta.copy()
            meta.pop("nodata", None)
        meta.update({"dtype":"int16", "count":2})

        # process the flow direction data
        fd_raster = fd_raster.astype("int16")
        direction_to_angle = {0:0, 255:0, 1:0, 2:45, 4:90, 8:135, 16:180, 32:225, 64:270, 128:315}
        fd_raster = np.vectorize(direction_to_angle.__getitem__)(fd_raster)
        fd_raster_cos = np.cos(np.deg2rad(fd_raster))*10000
        fd_raster_sin = np.sin(np.deg2rad(fd_raster))*10000

        if scale == "local": # remove the data where there is no AOI
            fd_raster_cos = np.where(reference_label == 0, 0, fd_raster_cos)
            fd_raster_sin = np.where(reference_label == 0, 0, fd_raster_sin)

        # save the final raster
        with rasterio.open(save_path, "w", **meta, compress="LZW") as file:
            file.write(fd_raster_cos, 1)
            file.set_band_description(1, "flow_direction_cos")
            file.write(fd_raster_sin, 2)
            file.set_band_description(2, "flow_direction_sin")
            file.nodata = None
        
        if os.path.exists(f"{save_path}.aux.xml"):
            os.remove(f"{save_path}.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create rasters for the flow direction data.")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, context, or basin.")

    args = parser.parse_args()

    create_flow_direction(args.data_folder, args.global_folder, args.scale)




    