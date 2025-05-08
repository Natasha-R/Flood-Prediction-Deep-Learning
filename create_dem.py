import geopandas as gpd
import os
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from tqdm import tqdm
import argparse

def main(fabdem_folder, save_dem_folder):

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
                with rasterio.open(f"{save_dem_folder}/temp_{index}.tif", "w", **out_meta) as file:
                    file.write(out_image)

        # combine all of the masked DEM images together to create one DEM representing the aoi polygon
        temp_tiles_paths = [f"{save_dem_folder}/temp_{index}.tif" for index in range(len(tiles))]
        temp_tiles = [rasterio.open(path) for path in temp_tiles_paths]
        merged_image, merged_transform = merge(temp_tiles)
        merged_meta = temp_tiles[0].meta.copy()
        merged_meta.update({"driver": "GTiff",
                            "height": merged_image.shape[1],
                            "width": merged_image.shape[2],
                            "transform": merged_transform,
                            "dtype": "int16",
                            "nodata": 0})
        with rasterio.open(f"{save_dem_folder}/dem_{aois.loc[idx, 'geometry_id']}.tif", "w", **merged_meta, compress="LZW") as file:
            file.write(merged_image)
        for file in temp_tiles:
            file.close()
        for path in temp_tiles_paths:
            os.remove(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create DEM files clipped to the extent of the AOIs.")
    parser.add_argument("--fabdem_folder", required=True, help="The path to the folder containing the full downloaded FABDEM.")
    parser.add_argument("--save_dem_folder", required=True, help="The path to the folder in which to save the DEM files clipped to the AOI extents.")
    args = parser.parse_args()

    main(fabdem_folder=args.fabdem_folder, save_dem_folder=args.save_dem_folder)


