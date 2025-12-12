import geopandas as gpd
from tqdm import tqdm
from osgeo import gdal
import os
import rasterio
import numpy as np
import argparse
import shapely
gdal.UseExceptions()

def create_flow_accumulation(data_folder, global_folder, scale):
    """
    Create rasters representing the flow accumulation data, corresponding to the CEMS label rasters.
    """

    # define the metadata and folder locations
    global_flow_accumulation = f"{global_folder}/global_flow_accumulation/global_flow_accumulation_3s.tif"
    if scale == "local":
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        flow_acc_folder = f"{data_folder}/full_subevent/raster_flow_accumulation"
    else: # if scale == "context" or scale == "basin"
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(shapely.wkt.loads)
        flow_acc_folder = f"{data_folder}/{scale}/flow_accumulation"
        if scale == "basin":
            global_flow_accumulation = f"{global_folder}/global_flow_accumulation/global_flow_accumulation_30s.tif"
    if not os.path.isdir(flow_acc_folder):
        os.mkdir(flow_acc_folder)

    for index in tqdm(range(len(raster_extents))):

        # import the metadata for the current raster
        bounds = raster_extents["geometry"][index].bounds
        height = int(raster_extents["height"].iloc[index])
        width = int(raster_extents["width"].iloc[index])
        subevent = raster_extents["subevent"][index]

        if scale == "local":
            save_path = f"{flow_acc_folder}/{subevent}.tif"
        else:  # if scale == "context" or scale == "basin"
            patch_name = raster_extents["patch"].iloc[index]
            save_path = f"{flow_acc_folder}/{patch_name}"

        if os.path.exists(save_path):
            continue

        # import the CEMS label as a reference to the extent of the AOIs
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)

        # extract the extent bounds from the global flow accumulation raster
        gdal.Warp(save_path, global_flow_accumulation, srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff',
                    resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)
        
        # scale the raster data values
        with rasterio.open(save_path) as file:
            fa_raster = file.read(1)
            meta = file.meta.copy()
            meta.pop("nodata", None)
        meta.update({"dtype":"uint16"})
        fa_raster = np.where(fa_raster == 4294967295, 0, fa_raster)
        fa_raster = np.log(fa_raster + 1)
        fa_raster = fa_raster * 1000

        # remove the data where there is no AOI
        if scale == "local":
            fa_raster = np.where(reference_label == 0, 0, fa_raster)

        # save the final raster
        with rasterio.open(save_path, "w", **meta, compress="LZW") as file:
            file.write(fa_raster, 1)
            file.set_band_description(1, "flow_accumulation")
            file.nodata = None
        
        if os.path.exists(f"{save_path}.aux.xml"):
            os.remove(f"{save_path}.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create rasters for the flow accumulation data.")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, context, or basin.")

    args = parser.parse_args()

    create_flow_accumulation(args.data_folder, args.global_folder, args.scale)
