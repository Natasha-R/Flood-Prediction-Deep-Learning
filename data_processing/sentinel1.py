import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import shape, GeometryCollection
from shapely import make_valid
import os
import rasterio
import math
import time
import numpy as np
from osgeo import gdal
import argparse
from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, BBoxSplitter, SentinelHubRequest, bbox_to_dimensions, CRS, MimeType, Geometry, MosaickingOrder)
gdal.UseExceptions()

def setup():
    config = SHConfig()
    config.sh_client_id = os.environ["COPERNICUS_CLIENT_ID"]
    config.sh_client_secret = os.environ["COPERNICUS_CLIENT_SECRET"]
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")
    config = SHConfig("cdse")
    return config

def find_sentinel1_availability(data_folder):
    """
    Use the Sentinel Hub catalog to find the availability and associated metadata of the Sentinel 1 data.
    The created "date difference" data are saved in geojson files.
    """

    config = setup()
    catalog = SentinelHubCatalog(config=config)

    # import the metadata
    aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    aois["one_week_previous"] = aois["event_date"] - pd.Timedelta(days=7)
    aois["180_days_previous"] = aois["event_date"] - pd.Timedelta(days=180)

    sentinel1_geojson_folder = f"{data_folder}/full_subevent/geojson_sentinel1/"
    if not os.path.isdir(sentinel1_geojson_folder):
        os.mkdir(sentinel1_geojson_folder)

    for index in tqdm(range(len(aois))):

        subevent = aois.loc[index, "subevent"]
        aoi_path = f"{sentinel1_geojson_folder}/{subevent}_aoi_{index}"

        # downloading the Sentinel 1 data will use the "MOST_RECENT" parameter within a given date range, but the downloaded data does not include the associated date
        # therefore we search the catalog for the same aoi in the same timeframe, to access the metadata for the sentinel 1 data availability
        search_iterator = catalog.search(DataCollection.SENTINEL1_IW,
                                        geometry=Geometry(aois.loc[index, "geometry"], CRS.WGS84),
                                        time=(aois.loc[index, "180_days_previous"], aois.loc[index, "one_week_previous"]),
                                        fields={"include": ["id", "properties.datetime", "properties.s1:polarization", "geometry"], "exclude": []})
        results = pd.DataFrame([{"id": item["id"], 
                                "datetime": item["properties"]["datetime"],
                                "polarization": item["properties"]["s1:polarization"],
                                "geometry":item["geometry"]} 
                                for item in list(search_iterator)])
        
        # format the response from the catalog API
        results["datetime"] = pd.to_datetime(results["datetime"]).dt.normalize()
        results = results[results["polarization"]=="DV"].sort_values("datetime", ascending=False).reset_index(drop=True)
        results["geometry"] = results["geometry"].apply(shape)
        results["orbit_number"] = results["id"].str.split("_").str[6]

        # find the date difference between the sentinel 1 capture date and the flood event date
        results["days_difference"] = (aois.loc[index, "aoi_date"] - results["datetime"].dt.tz_localize(None)).dt.days
        results = gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326")

        # the polygons corresponding to a sentinel 1 data granule are not exact, and sometimes leave gaps
        # calculate the convex hull between granules corresponding to the same orbit, and thus the same capture
        sentinel1_availability = {"days_differences":[], "geometry":[]}
        for (days_difference, orbit_number), data in results.groupby(["days_difference", "orbit_number"]):
            merged_polygons = unary_union(data["geometry"]).convex_hull
            sentinel1_availability["days_differences"].append(days_difference)
            sentinel1_availability["geometry"].append(merged_polygons)
        sentinel1_availability = gpd.GeoDataFrame(sentinel1_availability, crs="EPSG:4326")
        sentinel1_availability = sentinel1_availability.sort_values("days_differences", ascending=False, ignore_index=True)

        sentinel1_availability.to_file(aoi_path + ".geojson")

def download_sentinel1(data_folder):
    """
    Download Sentinel 1 data for each of the AOIs.
    """

    config = setup()

    # import the metadata and calculate the time frame for data collection
    aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    aois = aois.drop_duplicates(["geometry_event_date_id"])
    aois["one_week_previous"] = aois["event_date"] - pd.Timedelta(days=7)
    aois["180_days_previous"] = aois["event_date"] - pd.Timedelta(days=180)

    sentinel1_folder = f"{data_folder}/full_subevent/sentinel_1"
    if not os.path.isdir(sentinel1_folder):
        os.mkdir(sentinel1_folder)

    # download data for each aoi individually
    for index in tqdm(list(aois.index)):

        subevent = aois.loc[index, "subevent"]
        save_data_folder = f"{sentinel1_folder}/aoi_{aois.loc[index, "geometry_event_date_id"]}"

        if not os.path.isdir(save_data_folder):
            # print("\n", "\n", "Index:", index, "Folder:", save_data_folder)

            # split the aoi into boxes of a maximum 2500x2500 pixels each
            geometry = Geometry(aois.loc[index, "geometry"], CRS.WGS84)
            geometry_size = bbox_to_dimensions(geometry.bbox, resolution=10)
            col_num  = math.ceil(geometry_size[0] / 2500)
            row_num = math.ceil(geometry_size[1] / 2500)
            geometry_correctly_sized = False
            while not geometry_correctly_sized: 
                split_bboxes = BBoxSplitter(shape_list=[geometry],
                                            crs=CRS.WGS84,
                                            split_shape=(col_num, row_num),
                                            reduce_bbox_sizes=True)
                split_aois = split_bboxes.get_bbox_list()
                col_increase = False
                row_increase = False
                for split_aoi in split_aois:
                    split_aoi_size = bbox_to_dimensions(split_aoi, resolution=10)
                    if split_aoi_size[0] > 2500:
                        col_increase = True
                    if split_aoi_size[1] > 2500:
                        row_increase = True
                if col_increase or row_increase:
                    if col_increase:
                        col_num += 1
                    if row_increase:
                        row_num += 1
                else:
                    geometry_correctly_sized = True

            for bbox in tqdm(split_aois, leave=False):
                bbox_size = bbox_to_dimensions(bbox, resolution=10)

                # restrict the start of the data retrieval date interval to known Sentinel 1 data availability
                availability = gpd.read_file(f"{data_folder}/full_subevent/geojson_sentinel1/{subevent}_aoi_{index}.geojson").sort_values("days_differences", ascending=True)
                availability["geometry"] = availability.geometry.apply(lambda row: make_valid(row, method="structure"))
                claimed_area = GeometryCollection()
                records = []
                for _, row in availability.iterrows():
                    remainder = row.geometry.difference(claimed_area)
                    if remainder.is_empty:
                        continue
                    record = row.drop("geometry").to_dict()
                    record["geometry"] = remainder
                    records.append(record)
                    claimed_area = claimed_area.union(remainder)
                    claimed_area = make_valid(claimed_area, method="structure")
                flattened = gpd.GeoDataFrame(records, crs=availability.crs)
                filtered = flattened[flattened.intersects(bbox.geometry)]
                start_interval = aois.loc[index, "aoi_date"] - pd.Timedelta(days=filtered["days_differences"].max())

                # download sentinel 1 data (bands VV and VH)
                request = SentinelHubRequest(
                    evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{
                                bands: ["VV", "VH"],
                            }],
                            output: {
                                bands: 2,
                                sampleType: "UINT16"
                            }
                        };
                    }
                    function toDb(linear) {
                        var log = 10 * Math.log(linear) / Math.LN10
                        var db = Math.max(0, (log + 20) / 30)
                        var scaled = db * 10000
                        return scaled
                    }
                    function evaluatePixel(sample) {
                        return [toDb(sample.VV), toDb(sample.VH)];
                    }
                """,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL1_IW.define_from("s1iw", service_url="https://sh.dataspace.copernicus.eu"),
                            time_interval=(start_interval, aois.loc[index, "one_week_previous"]),
                            mosaicking_order=MosaickingOrder.MOST_RECENT,
                            other_args={"dataFilter":{"resolution":"HIGH","acquisitionMode":"IW", "polarization":"DV", "mosaickingOrder": "mostRecent"},
                                        "processing":{"backCoeff":"GAMMA0_TERRAIN","orthorectify":True,"demInstance":"COPERNICUS", "upsampling":"BILINEAR"}})],

                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=bbox,
                    size=bbox_size,
                    geometry=geometry,
                    config=config,
                    data_folder=save_data_folder)
                
                data = request.get_data(save_data=True, redownload=False)

                time.sleep(0.5)

def create_sentinel1_aoi_date_difference(data_folder):
    """
    Create rasters of the Sentinel 1 data for each individual AOI.
    Add a band to the raster representing the time difference between Sentinel 1 data capture and the subevent date.
    """

    # import the metadata
    aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    aois["one_week_previous"] = aois["event_date"] - pd.Timedelta(days=7)
    aois["180_days_previous"] = aois["event_date"] - pd.Timedelta(days=180)

    sentinel1_geojson_folder = f"{data_folder}/full_subevent/geojson_sentinel1/"
    sentinel1_raster_folder = f"{data_folder}/full_subevent/raster_sentinel1/"
    if not os.path.isdir(sentinel1_raster_folder):
        os.mkdir(sentinel1_raster_folder)

    for index in tqdm(range(len(aois))):

        aoi_id = aois.loc[index, 'geometry_event_date_id']
        subevent = aois.loc[index, "subevent"]
        aoi_path = f"{sentinel1_raster_folder}/{subevent}_aoi_{index}"
        geojson_path = f"{sentinel1_geojson_folder}/{subevent}_aoi_{index}"

        # some of the larger AOIs needed to be split up in order to download them. Join them back together to form the full AOI
        sub_aoi_paths = [os.path.join(root, file) for root, dirs, files in os.walk(f"{data_folder}/full_subevent/sentinel_1/aoi_{aoi_id}") for file in files if file.endswith(".tiff")]
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        gdal.Warp(f"{aoi_path}.tif", sub_aoi_paths, srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', 
                  outputType=gdal.GDT_Int16, resampleAlg="bilinear")
        gdal.PopErrorHandler()

        # import the sentinel 1 data for the aoi and its associated metadata
        with rasterio.open(aoi_path + ".tif") as s1_file:
            bounds = tuple(s1_file.bounds)
            height = s1_file.height
            width = s1_file.width
            vv_data = s1_file.read(1)
            vh_data = s1_file.read(2)
            meta = s1_file.meta.copy()
            meta.update(count=3)

        # convert the geojson corresponding to the sentinel 1 date difference metadata into raster format
        gdal.Rasterize(aoi_path + "_date_diff.tif", geojson_path + ".geojson", format="GTiff", attribute="days_differences", 
                       width=width, height=height, outputBounds=bounds, 
                       outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=LZW"])
        with rasterio.open(aoi_path + "_date_diff.tif") as date_diff_file:
            date_difference = date_diff_file.read(1)

        # create a raster representing the AOI, containing bands for the VV and VH data, and the date difference metadata as a layer
        with rasterio.open(aoi_path + ".tif", 'w', **meta, compress="LZW") as file:
            file.write(vv_data, 1)
            file.write(vh_data, 2)
            file.write(date_difference, 3)
            file.set_band_description(1, "VV")
            file.set_band_description(2, "VH")
            file.set_band_description(3, "date_difference")

        time.sleep(0.5)
        os.remove(aoi_path + "_date_diff.tif")

def create_sentinel1_rasters(data_folder):
    """
    Combine the Sentinel 1 individual AOI rasters into rasters for the full subevent, corresponding to the CEMS labels.
    """
    
    raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
    sentinel1_raster_folder = f"{data_folder}/full_subevent/raster_sentinel1/"

    for index in tqdm(range(len(raster_extents))):

        # extract the metadata corresponding to each subevent raster
        subevent = raster_extents["subevent"][index]
        raster_bounds = raster_extents["geometry"][index].bounds
        raster_height = int(raster_extents["height"].iloc[index])
        raster_width = int(raster_extents["width"].iloc[index])

        # for each of the aois contained within a particular subevent raster, join them together to form one full raster for sentinel 1 data
        aois_in_raster = [f"{sentinel1_raster_folder}/{file}" for file in os.listdir(sentinel1_raster_folder) if subevent in file]
        gdal.Warp(f"{sentinel1_raster_folder}/{subevent}.tif", aois_in_raster, 
                  srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', outputType=gdal.GDT_Int16,
                  resampleAlg="bilinear", width=raster_width, height=raster_height, outputBounds=raster_bounds)
        
        # set the date difference band to 0 outside of the AOI bounds
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)
        with rasterio.open(f"{sentinel1_raster_folder}/{subevent}.tif") as sentinel1_file:
            sentinel1_raster = sentinel1_file.read()
            meta = sentinel1_file.meta.copy()
            count = sentinel1_file.count
            raster_height, raster_width = sentinel1_file.height, sentinel1_file.width
        sentinel1_raster[count-1] = np.where(reference_label == 0, 0, sentinel1_raster[count-1])

        # add descriptions to the bands and save the final raster for the sentinel 1 data
        band_descriptions = ["VV", "VH", "date_difference"]
        with rasterio.open(f"{sentinel1_raster_folder}/{subevent}.tif", 'w', **meta, compress="LZW") as file:
            for band_index in range(count):
                file.write(sentinel1_raster[band_index], band_index + 1)
                file.set_band_description(band_index + 1, band_descriptions[band_index])

    # remove the separate aoi tiff files
    aoi_tiffs = [os.path.join(root, file) for root, dirs, files in os.walk(f"{sentinel1_raster_folder}") for file in files if "aoi" in file]
    for tiff in aoi_tiffs:
        os.remove(tiff)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download Sentinel 1 data and create rasters")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--find_sentinel1_availability", action="store_true", default=False, help="Create metadata describing the availability of Sentinel 2 data.")
    parser.add_argument("--download_sentinel1", action="store_true", default=False, help="Download the Sentinel 1 data for each of the AOIs.")
    parser.add_argument("--create_sentinel1_aoi_date_difference", action="store_true", default=False, help="Create a raster for each AOI and add a date difference band to it.")
    parser.add_argument("--create_sentinel1_rasters", action="store_true", default=False, help="Create raster files of the Sentinel 1 data, matching to the CEMS labels.")

    args = parser.parse_args()

    if args.find_sentinel1_availability:
        find_sentinel1_availability(args.data_folder)

    if args.download_sentinel1:
        download_sentinel1(args.data_folder)
    
    if args.create_sentinel1_aoi_date_difference:
        create_sentinel1_aoi_date_difference(args.data_folder)
    
    if args.create_sentinel1_rasters:
        create_sentinel1_rasters(args.data_folder)