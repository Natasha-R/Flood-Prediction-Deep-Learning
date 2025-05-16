import geopandas as gpd
import os
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from tqdm import tqdm
import argparse
from osgeo import gdal
gdal.UseExceptions()

def extract_dem(fabdem_folder, extract_dem_folder):

    # import the aoi and DEM metadata
    fabdem = gpd.read_file("metadata/FABDEM_tiles.geojson").to_crs(epsg=4326)
    aois = gpd.read_file("metadata/aoi_extent.geojson")
    aois = aois.drop_duplicates(["geometry_id"], ignore_index=True)

    for idx in tqdm(range(len(aois))):

        # extract which DEM tiles are covered by the aoi polygon
        geometry = aois.loc[idx, "geometry"]
        tiles = list(fabdem[fabdem.intersects(geometry)]["file_name"])
        tiles = [f"{fabdem_folder}/{tile[0]}{tile[2:]}" for tile in tiles]

        # for each DEM tile, extract data only within the aoi polygon, and save temporarily
        for index, tile in enumerate(tiles):
            with rasterio.open(tile) as file:
                out_image, out_transform = mask(file, [geometry], crop=True, nodata=0)
                out_meta = file.meta.copy()
                out_meta.update({"driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform,
                                "nodata": 0})
                with rasterio.open(f"{extract_dem_folder}/temp_{index}.tif", "w", **out_meta) as file:
                    file.write(out_image)

        # combine all of the masked DEM images together to create one DEM representing the aoi polygon
        temp_tiles_paths = [f"{extract_dem_folder}/temp_{index}.tif" for index in range(len(tiles))]
        temp_tiles = [rasterio.open(path) for path in temp_tiles_paths]
        merged_image, merged_transform = merge(temp_tiles)
        merged_meta = temp_tiles[0].meta.copy()
        merged_meta.update({"driver": "GTiff",
                            "height": merged_image.shape[1],
                            "width": merged_image.shape[2],
                            "transform": merged_transform,
                            "dtype": "int16",
                            "nodata": 0})
        with rasterio.open(f"{extract_dem_folder}/dem_{aois.loc[idx, 'geometry_id']}.tif", "w", **merged_meta, compress="LZW") as file:
            file.write(merged_image)
        for file in temp_tiles:
            file.close()
        for path in temp_tiles_paths:
            os.remove(path)

def create_dem(extract_dem_folder, create_dem_folder):

    # import in the metadata on all of the AOIs and rasters
    aois = gpd.read_file("metadata/aoi_extent.geojson")
    rasters = gpd.read_file("metadata/raster_extent.geojson")
    rasters["raster_geometry"] = rasters["geometry"]
    aois = aois.merge(rasters[["subevent", "height", "width", "raster_geometry"]], how="left", on="subevent")
    
    for subevent, data in tqdm(aois.groupby("subevent")):

        # determine which DEM AOIs need to be combined, and the final extent of the raster
        dem_files = list(data["geometry_id"].apply(lambda row: f"{extract_dem_folder}/dem_{row}.tif"))
        height = int(data["height"].iloc[0])
        width = int(data["width"].iloc[0])
        bounds = data["raster_geometry"].iloc[0].bounds

        # create the DEM raster that matches to the CEMS label raster
        gdal.Warp(f"{create_dem_folder}/{subevent}.tif", dem_files, 
                  srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', outputType=gdal.GDT_Int16, creationOptions=["COMPRESS=LZW"],
                  width=width, height=height, outputBounds=bounds)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create DEM files clipped to the extent of the AOIs.")
    parser.add_argument("--fabdem_folder", default=None, help="The path to the folder containing the full downloaded FABDEM.")
    parser.add_argument("--extract_dem", action="store_true", default=False, help="Extract the AOIs from the FABDEM.")
    parser.add_argument("--extract_dem_folder", default=None, help="The path to the folder in which to save the DEM files clipped to the AOI extents.")
    parser.add_argument("--create_dem_folder", default=None, help="The path to the folder in which to save the final DEM rasters.")
    parser.add_argument("--create_dem", action="store_true", default=False, help="Save the extracted DEM AOIs as rasters matching the CEM labels.")
    
    args = parser.parse_args()

    if args.extract_dem:
        extract_dem(fabdem_folder=args.fabdem_folder, extract_dem_folder=args.extract_dem_folder)

    if args.create_dem:
        create_dem(extract_dem_folder=args.extract_dem_folder, create_dem_folder=args.create_dem_folder)
    


