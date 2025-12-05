import geopandas as gpd
import os
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from tqdm import tqdm
import argparse
from osgeo import gdal
import shutil
import numpy as np
import shapely
from rasterio.enums import Resampling
gdal.UseExceptions()

def extract_dem_aoi(data_folder, global_folder):
    """
    Extract the DEM data from FABDEM for each of the AOIs.
    """

    # import the aoi and DEM metadata
    fabdem = gpd.read_file(f"{global_folder}/global_fabdem/FABDEM_v1-2_tiles.geojson").to_crs(epsg=4326)
    aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    aois = aois.drop_duplicates(["geometry_id"], ignore_index=True)

    dem_aoi_folder = f"{data_folder}/full_subevent/raster_dem_aoi"
    if not os.path.isdir(dem_aoi_folder):
        os.mkdir(dem_aoi_folder)

    for idx in tqdm(range(len(aois))):

        # extract which DEM tiles are covered by the aoi polygon
        geometry = aois.loc[idx, "geometry"]
        tiles = list(fabdem[fabdem.intersects(geometry)]["file_name"])
        tiles = [f"{global_folder}/global_fabdem/{tile[0]}{tile[2:]}" for tile in tiles]

        # for each DEM tile, extract data only within the aoi polygon, and save temporarily
        for index, tile in enumerate(tiles):
            with rasterio.open(tile) as file:
                out_image, out_transform = mask(file, [geometry], crop=True)
                out_meta = file.meta.copy()
                out_meta.update({"driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform})
                with rasterio.open(f"{dem_aoi_folder}/temp_{index}.tif", "w", **out_meta) as file:
                    file.write(out_image)

        # combine all of the masked DEM images together to create one DEM representing the aoi polygon
        temp_tiles_paths = [f"{dem_aoi_folder}/temp_{index}.tif" for index in range(len(tiles))]
        temp_tiles = [rasterio.open(path) for path in temp_tiles_paths]
        merged_image, merged_transform = merge(temp_tiles)
        merged_meta = temp_tiles[0].meta.copy()
        merged_meta.update({"driver": "GTiff",
                            "height": merged_image.shape[1],
                            "width": merged_image.shape[2],
                            "transform": merged_transform,
                            "dtype": "int16",
                            "nodata":0,
                            "resampling": Resampling.bilinear})
        merged_image = np.where(merged_image == -9999, 0, merged_image)
        merged_image = np.round(merged_image).astype(np.int16)
        with rasterio.open(f"{dem_aoi_folder}/dem_{aois.loc[idx, 'geometry_id']}.tif", "w", **merged_meta, compress="LZW") as file:
            file.write(merged_image)
            file.set_band_description(1, "DEM")
            file.nodata = 0
        for file in temp_tiles:
            file.close()
        for path in temp_tiles_paths:
            os.remove(path)

def create_dem_rasters(data_folder):
    """
    Create rasters of the DEM data, corresponding to the CEMS label rasters.
    """
    
    # import in the metadata on all of the AOIs and rasters
    aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    rasters = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    rasters["raster_geometry"] = rasters["geometry"]
    aois = aois.merge(rasters[["subevent", "height", "width", "raster_geometry"]], how="left", on="subevent")

    dem_aoi_folder = f"{data_folder}/full_subevent/raster_dem_aoi"
    dem_raster_folder = f"{data_folder}/full_subevent/raster_dem"
    if not os.path.isdir(dem_raster_folder):
        os.mkdir(dem_raster_folder)

    for subevent, data in tqdm(aois.groupby("subevent")):

        # determine which DEM AOIs need to be combined, and the final extent of the raster
        dem_files = list(data["geometry_id"].apply(lambda row: f"{dem_aoi_folder}/dem_{row}.tif"))
        height = int(data["height"].iloc[0])
        width = int(data["width"].iloc[0])
        bounds = data["raster_geometry"].iloc[0].bounds

        # create the DEM raster that matches to the CEMS label raster
        gdal.Warp(f"{dem_raster_folder}/{subevent}.tif", dem_files, 
                  srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', outputType=gdal.GDT_Int16, creationOptions=["COMPRESS=LZW"],
                  resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)
        
    # remove the intermediary files
    shutil.rmtree(dem_aoi_folder)

def create_dem_scales(data_folder, global_folder, scale):
    """
    Create patches of the DEM data at the context and basin scales.
    """

    # define the metadata and the folder locations
    fabdem = gpd.read_file(f"{global_folder}/global_fabdem/FABDEM_v1-2_tiles.geojson").to_crs(epsg=4326)
    scales = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
    scales["geometry"] = scales[f"{scale}_geometry"].apply(shapely.wkt.loads)
    dem_folder = f"{data_folder}/{scale}/dem"
    if not os.path.isdir(dem_folder):
        os.mkdir(dem_folder)

    for index in tqdm(range(len(scales))):

        # import the metadata for the patch
        height = int(scales["height"].iloc[index])
        width = int(scales["width"].iloc[index])
        geometry = scales["geometry"].iloc[index]
        bounds = scales["geometry"][index].bounds
        patch_name = scales["patch"].iloc[index]

        # extract all of the DEM tiles that intersect with the patch geometry
        tiles = list(fabdem[fabdem.intersects(geometry)]["file_name"])
        tiles = [f"{global_folder}/global_fabdem/{tile[0]}{tile[2:]}" for tile in tiles]
        all_idx = []
        for idx, tile in enumerate(tiles):
            with rasterio.open(tile) as file:
                if shapely.geometry.box(*file.bounds).intersects(geometry):
                    out_image, out_transform = mask(file, [geometry], crop=True)
                    out_meta = file.meta.copy()
                    out_meta.update({"driver": "GTiff", "dtype": "int16", "transform": out_transform,
                                    "height": out_image.shape[1], "width": out_image.shape[2]})
                    out_meta.pop("nodata", None)
                    out_image = np.where(out_image == -9999, 0, out_image)
                    out_image = np.round(out_image).astype(np.int16)
                    with rasterio.open(f"{dem_folder}/temp_{idx}.tif", "w", **out_meta) as file:
                        file.write(out_image)
                        file.nodata = None
                    all_idx.append(idx)

        # combine all of the individual DEM tiles
        temp_tiles_paths = [f"{dem_folder}/temp_{idx}.tif" for idx in all_idx]
        gdal.Warp(f"{dem_folder}/{patch_name}", temp_tiles_paths, 
                  srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', outputType=gdal.GDT_Int16, creationOptions=["COMPRESS=LZW"],
                  resampleAlg="bilinear", width=width, height=height, outputBounds=bounds)
        with rasterio.open(f"{dem_folder}/{patch_name}") as file:
            raster = file.read(1)
            meta = file.meta.copy()
        with rasterio.open(f"{dem_folder}/{patch_name}", "w", **meta, compress="LZW") as file:
            file.write(raster, 1)
            file.set_band_description(1, "DEM")
        
        for path in temp_tiles_paths:
            os.remove(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create DEM files clipped to the extent of the AOIs, and save in rasters matching to the CEMS labels.")
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--global_folder", default=os.environ["GLOBAL_FOLDER"], help="The path to the folder containing the global data")
    parser.add_argument("--extract_dem_aoi", action="store_true", default=False, help="Extract the DEM for each of the AOIs from the FABDEM files.")
    parser.add_argument("--create_dem_rasters", action="store_true", default=False, help="Create raster files for the DEM, matching to the CEMS labels.")
    parser.add_argument("--create_dem_scales", action="store_true", default=False, help="Create patches of the DEM data at the context and basin scales.")
    parser.add_argument("--scale", default="context", help="The scale at which to create raster files: context or basin.")

    args = parser.parse_args()

    if args.extract_dem_aoi:
        extract_dem_aoi(data_folder=args.data_folder, global_folder=args.global_folder)

    if args.create_dem_rasters:
        create_dem_rasters(data_folder=args.data_folder)

    if args.create_dem_scales:
        create_dem_scales(args.data_folder, args.global_folder, args.scale)