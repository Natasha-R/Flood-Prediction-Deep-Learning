import geopandas as gpd
import os
import argparse
from creds import *
from tqdm import tqdm
import time
import math
from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, BBoxSplitter, bbox_to_dimensions, CRS, MimeType, Geometry, MosaickingOrder)
import pandas as pd
import numpy as np
from osgeo import gdal
import rasterio
from shapely import wkt
from shapely.geometry import shape
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import shutil
gdal.UseExceptions()
pd.set_option('mode.chained_assignment', None)

def setup():
    """
    Configure the connection to the Sentinel Hub API
    """
    config = SHConfig()
    config.sh_client_id = os.environ["COPERNICUS_CLIENT_ID"]
    config.sh_client_secret = os.environ["COPERNICUS_CLIENT_SECRET"]
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")
    config = SHConfig("cdse")
    return config

def find_sentinel2_availability(data_folder, scale):
    """
    Use the Sentinel Hub catalog to find the availability and associated metadata of Sentinel 2 data.
    """

    config = setup()
    catalog = SentinelHubCatalog(config=config)
        
    # import the metadata
    if scale == "local":
        aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    else:
        aois = gpd.read_file(f"{data_folder}/metadata/scales_aois.geojson")
        aois["geometry"] = aois[f"{scale}_geometry"].apply(wkt.loads)
        aois["event"] = aois["subevent"].str.split("_").str[0]
    aois = aois.drop_duplicates(["geometry_event_date_id"], ignore_index=True)
    aois["earlier_date"] = aois["event_date"] - pd.Timedelta(days=90)

    # get metadata on available sentinel 2 data with its id, time, and cloud cover for the given event and date
    availability = []
    for index in tqdm(range(len(aois)), desc=f"Find Sentinel 2 availability at {scale} scale"):

        search_iterator = catalog.search(
            DataCollection.SENTINEL2_L2A,
            geometry=Geometry(aois.loc[index, "geometry"], CRS.WGS84),
            time=(aois.loc[index, "earlier_date"], aois.loc[index, "event_date"]),
            fields={"include": ["id", "geometry", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []})
        
        results = pd.DataFrame([{"id": item["id"], 
                                "datetime": item["properties"]["datetime"], 
                                "tile_geometry": shape(item["geometry"]),
                                "cloud_cover": item["properties"]["eo:cloud_cover"]} for item in list(search_iterator)])
        
        for attribute in ["event", "event_date", "geometry", "geometry_event_date_id"]:
            results[attribute] = aois.loc[index, attribute]

        availability.append(results)

        time.sleep(0.5)

    # process the results
    availability = pd.concat(availability, ignore_index=True)
    availability["version"] = availability["id"].str.split("_").str[3]
    availability["tile"] = availability["id"].str.split("_").str[5]
    availability["tile_date"] = availability["id"].str.split("_").str[2]
    availability = availability.sort_values("version", ascending=False)
    availability = availability.drop_duplicates(subset=["event", "event_date", "tile", "tile_date", "geometry_event_date_id"], keep="first")
    availability = availability.sort_values(["event", "geometry_event_date_id", "tile", "tile_date"], ignore_index=True)

    # save the availability results
    availability.to_csv(f"{data_folder}/metadata/sentinel2_{scale}_availability.csv", index=False) 

def find_minimal_cloud_cover(data_folder, scale):
    """
    Process the Sentinel 2 availability metadata to determine which data granules have the minimal cloud cover.
    """

    # import the metadata
    if scale == "local":
        aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
    else:
        aois = gpd.read_file(f"{data_folder}/metadata/scales_aois.geojson")
        aois["geometry"] = aois[f"{scale}_geometry"].apply(wkt.loads)
        aois["event"] = aois["subevent"].str.split("_").str[0]
        aois["earlier_date"] = aois["event_date"] - pd.Timedelta(days=90)
    availability = pd.read_csv(f"{data_folder}/metadata/sentinel2_{scale}_availability.csv")

    # find the optimal date for Sentinel 2 data for each aoi, based on the cloud cover percentages
    cloud_cover_availability = {"geometry_event_date_id":[], "aoi_group": [], "availability_date":[], "cloud_cover_percentage":[]}
    for geometry_event_date_id, data in tqdm(availability.groupby("geometry_event_date_id"), desc=f"Find optimal cloud cover from availability metadata at {scale} scale"):

        # only use dates for which data is available within the entire aoi
        coverage_by_date = data.groupby(["tile_date"])["tile_geometry"].apply(lambda row : unary_union(wkt.loads(row)).convex_hull)
        aoi_geometry = wkt.loads(list(data["geometry"])[0])
        full_coverage_available = coverage_by_date.apply(lambda tile_geometry: tile_geometry.contains(aoi_geometry))
        full_coverage_available = list(full_coverage_available[full_coverage_available].index)

        # if there are no dates on which data is available within the entire aoi, multiple dates must be used
        if len(full_coverage_available) == 0:

            # the different orbit captures can be grouped by the amount of area captured by that orbit
            coverage_amount = coverage_by_date.apply(lambda tile_geometry: (tile_geometry.intersection(aoi_geometry).area / aoi_geometry.area)*100)

            # find the number of different orbits/groups of captures, then group them
            coverage_array = coverage_amount.values.reshape(-1, 1)
            best_k, best_score = 2, -1
            for k in range(2, min(10, len(coverage_amount))):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(coverage_array)
                score = silhouette_score(coverage_array, kmeans.labels_)
                if score > best_score:
                    best_k, best_score = k, score
            kmeans = KMeans(n_clusters=best_k, random_state=0).fit(coverage_array)
            clustered_captures = pd.DataFrame({"tile_date": coverage_amount.index, "area_covered": coverage_amount.values, "aoi_group": kmeans.labels_})

            # the larger the group number, the more area it covers, hence the priority it should have in composites
            sorted_groups = clustered_captures.groupby("aoi_group")["area_covered"].mean().sort_values().index.tolist()
            clustered_captures["aoi_group"] = clustered_captures["aoi_group"].map({old_group_name: new_group_name for new_group_name, old_group_name in enumerate(sorted_groups)})

            data = data.merge(clustered_captures[["tile_date", "aoi_group"]], on="tile_date")

        else:
            data = data[data["tile_date"].isin(full_coverage_available)]
            data["aoi_group"] = 0

        # find the optimum cloud cover for each individual aoi group
        for aoi_group, aoi_group_data in data.groupby("aoi_group"):

            event_date = pd.to_datetime(aoi_group_data["event_date"].head(1).item())
            aoi_group_data["tile_date"] = pd.to_datetime(aoi_group_data["tile_date"])

            # find the average cloud cover across all of the tiles, per date
            cloud_cover_means = aoi_group_data.groupby(["tile_date"])["cloud_cover"].mean().reset_index()

            # the optimum date, based on the amount of cloud cover, fits the following criteria:
            # if cloud cover is under 5% within 30 days of the flood event, or if cloud cover is under 10% within 30 days of the flood event
            # if this criteria isn't met, then complete the same search within 60 days, and then 90 days
            # if this criteria isn't met, then simply return the date with the lowest cloud cover percentage.
            selected_cloud_cover = pd.DataFrame()
            for window in [30, 60, 90]:
                cloud_cover_means_within_window = cloud_cover_means[cloud_cover_means["tile_date"] > event_date - pd.Timedelta(days=window)]
                for threshold in [5, 10]:
                    result = cloud_cover_means_within_window[cloud_cover_means_within_window["cloud_cover"] < threshold].sort_values("tile_date", ascending=False)
                    if not result.empty:
                        selected_cloud_cover = result
                        break
                if not selected_cloud_cover.empty:
                    break
            if selected_cloud_cover.empty:
                selected_cloud_cover = cloud_cover_means.sort_values("cloud_cover", ascending=True)

            cloud_cover_availability["geometry_event_date_id"].append(geometry_event_date_id)
            cloud_cover_availability["aoi_group"].append(aoi_group)
            cloud_cover_availability["availability_date"].append(selected_cloud_cover.head(1)["tile_date"].item())
            cloud_cover_availability["cloud_cover_percentage"].append(selected_cloud_cover.head(1)["cloud_cover"].item())

    cloud_cover_availability = pd.DataFrame(cloud_cover_availability)
    aois = aois.merge(cloud_cover_availability, on="geometry_event_date_id")

    aois.to_file(f"{data_folder}/metadata/s2_{scale}_aoi_availability.geojson")

def download_sentinel2(data_folder, scale):
    """
    Download the Sentinel 2 data for each of the AOIs.
    """

    # configure the API credentials
    config = setup()

    # import the aoi data with the Sentinel 2 availability
    aois = gpd.read_file(f"{data_folder}/metadata/s2_{scale}_aoi_availability.geojson")
    aois = aois.drop_duplicates(["geometry_event_date_id", "aoi_group"], ignore_index=True)
    aois["availability_date"] = aois["availability_date"].dt.date
    if scale == "local":
        sentinel2_folder = f"{data_folder}/full_subevent/sentinel_2/"
    else:
        sentinel2_folder = f"{data_folder}/{scale}/download_sentinel2/"
    if not os.path.isdir(sentinel2_folder):
        os.mkdir(sentinel2_folder)
    resolution = {"local":10, "nearby":25, "context":100, "con_context":100, "basin":1000}[scale]

    for index in tqdm(range(len(aois)), desc=f"Download Sentinel 2 data for {scale} scale"):

        save_data_folder = f"{sentinel2_folder}/aoi_{aois.loc[index, 'geometry_event_date_id']}/group_{aois.loc[index, 'aoi_group']}/"
        if not os.path.isdir(save_data_folder):

            # split the aoi into boxes of a maximum 2500x2500 pixels each
            geometry = Geometry(aois.loc[index, "geometry"], CRS.WGS84)
            geometry_size = bbox_to_dimensions(geometry.bbox, resolution=resolution)
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
                    split_aoi_size = bbox_to_dimensions(split_aoi, resolution=resolution)
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

            # extract the sentinel 2 bands and save as a GeoTiff in UINT16
            for bbox in tqdm(split_aois):
                bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
                request = SentinelHubRequest(
                    evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{
                                bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "CLD"],
                                units: ["DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "DN", "PERCENT"]
                            }],
                            output: {
                                bands: 11,
                                sampleType: "UINT16"
                            }
                        };
                    }
                    function evaluatePixel(sample) {
                        return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, 
                                sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12, sample.CLD];
                    }
                """,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL2_L2A.define_from(name="s2", service_url="https://sh.dataspace.copernicus.eu"),
                            time_interval=(aois.loc[index, "availability_date"], aois.loc[index, "availability_date"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)),
                            mosaicking_order=MosaickingOrder.MOST_RECENT
                            )],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=bbox,
                    size=bbox_size,
                    geometry=geometry,
                    config=config,
                    data_folder=save_data_folder)
                
                data = request.get_data(save_data=True, redownload=False)

                time.sleep(0.5)
    
def set_sentinel2_nodata(data_folder, scale):
    """
    Set the nodata value to 0 in the downloaded sentinel 2 data.
    """
    if scale == "local":
        sentinel2_folder = f"{data_folder}/full_subevent/sentinel_2/"
    else:
        sentinel2_folder = f"{data_folder}/{scale}/download_sentinel2"
    all_responses = [os.path.join(root, file) for root, dirs, files in os.walk(sentinel2_folder) for file in files if file.endswith(".tiff")]
    for response_path in all_responses:
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            gdal.Translate(destName=response_path[:-5] + "_temp.tiff", srcDS=response_path, noData=0, creationOptions=["COMPRESS=LZW"])
            gdal.PopErrorHandler()
            shutil.move(response_path[:-5] + "_temp.tiff", response_path)

def create_sentinel2_rasters(data_folder, scale):
    """
    Combine the Sentinel 2 individual AOI rasters into rasters for the full subevent, corresponding to the CEMS labels.
    """

    # import the metadata
    aoi_avail = gpd.read_file(f"{data_folder}/metadata/s2_{scale}_aoi_availability.geojson")
    if scale == "local":
        aois = gpd.read_file(f"{data_folder}/metadata/aoi_extent.geojson")
        raster_extents = gpd.read_file(f"{data_folder}/metadata/raster_extent.geojson")
        sentinel2_raster_folder = f"{data_folder}/full_subevent/raster_sentinel2/"
        download_s2_folder = f"{data_folder}/full_subevent/sentinel_2"
    else:
        aois = gpd.read_file(f"{data_folder}/metadata/scales_aois.geojson")
        aois["geometry"] = aois[f"{scale}_geometry"].apply(wkt.loads)
        aoi_avail["aoi_date"] = aoi_avail["date"]
        raster_extents = gpd.read_file(f"{data_folder}/metadata/scales.geojson")
        raster_extents["geometry"] = raster_extents[f"{scale}_geometry"].apply(wkt.loads)
        sentinel2_raster_folder = f"{data_folder}/{scale}/sentinel2/"
        download_s2_folder =  f"{data_folder}/{scale}/download_sentinel2/"
    aoi_avail["availability_date"] = aoi_avail["availability_date"].dt.normalize()
    aoi_avail["date_difference"] = aoi_avail["aoi_date"]-aoi_avail["availability_date"]
    aoi_avail["date_difference"] = aoi_avail["date_difference"].dt.days
    if not os.path.isdir(sentinel2_raster_folder):
        os.mkdir(sentinel2_raster_folder)

    for index in tqdm(range(len(raster_extents)), desc=f"Create Sentinel 2 rasters for {scale} scale"):

        # extract the metadata
        subevent = raster_extents["subevent"][index]
        raster_bounds = raster_extents["geometry"][index].bounds
        raster_height = int(raster_extents["height"].iloc[index])
        raster_width = int(raster_extents["width"].iloc[index])
        if scale == "local":
            aoi_ids = list(aois[aois["subevent"]==subevent]["geometry_event_date_id"])
            raster_id = subevent
        else:
            aoi_ids = [raster_extents.loc[index, "geometry_event_date_id"]]
            raster_id = raster_extents["patch"].iloc[index][:-4]

        if os.path.isfile(f"{sentinel2_raster_folder}/{raster_id}.tif"):
            continue

        all_aoi_paths = []

        for aoi_id in aoi_ids:

            aoi_groups = os.listdir(f"{download_s2_folder}/aoi_{aoi_id}")
            aoi_groups.sort()

            # if an AOI could only be captured across multiple days, then each is in a separate group
            for aoi_group in aoi_groups:

                # some of the larger AOIs needed to be split up to download them. Join them back together to form the full AOI
                sub_aoi_paths = [os.path.join(root, file) for root, dirs, files in os.walk(f"{download_s2_folder}/aoi_{aoi_id}/{aoi_group}") for file in files if file.endswith(".tiff")]
                gdal.PushErrorHandler('CPLQuietErrorHandler')
                gdal.Warp(f"{sentinel2_raster_folder}/{raster_id}_{aoi_id}_{aoi_group}.tif", sub_aoi_paths, 
                        srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', outputType=gdal.GDT_Int16, resampleAlg="bilinear")
                gdal.PopErrorHandler()
                
                # add a date difference band to the AOI raster
                with rasterio.open(f"{sentinel2_raster_folder}/{raster_id}_{aoi_id}_{aoi_group}.tif") as sentinel2_file:
                    sentinel2_raster = sentinel2_file.read()
                    meta = sentinel2_file.meta.copy()
                    count = sentinel2_file.count
                    aoi_height, aoi_width = sentinel2_file.height, sentinel2_file.width
                    meta.update(count = count + 1)
                date_difference_days = aoi_avail[(aoi_avail["subevent"]==subevent) & (aoi_avail["geometry_event_date_id"]==aoi_id) & (aoi_avail["aoi_group"]==int(aoi_group.split("_")[-1]))].reset_index(drop=True)["date_difference"][0]
                date_difference = np.ones((aoi_height, aoi_width), dtype="uint16") * date_difference_days
                date_difference[np.all(sentinel2_raster == 0, axis=0)] = 0
                with rasterio.open(f"{sentinel2_raster_folder}/{raster_id}_{aoi_id}_{aoi_group}.tif", 'w', **meta) as file:
                    for band_index in range(count):
                        file.write(sentinel2_raster[band_index], band_index + 1)
                        file.write(date_difference, count + 1)

                all_aoi_paths.append(f"{sentinel2_raster_folder}/{raster_id}_{aoi_id}_{aoi_group}.tif")

        # join all of the AOI rasters together to form the overall subevent raster
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        gdal.Warp(f"{sentinel2_raster_folder}/{raster_id}.tif", all_aoi_paths, 
            srcSRS="EPSG:4326", dstSRS="EPSG:4326", format='GTiff', outputType=gdal.GDT_Int16,
            resampleAlg="bilinear", srcNodata=0, dstNodata=0, width=raster_width, height=raster_height, outputBounds=raster_bounds)
        gdal.PopErrorHandler()
            
        # set the date difference band to 0 where there is no AOI
        with rasterio.open(f"{data_folder}/full_subevent/raster_cems/{subevent}.tif") as reference_file:
            reference_label = reference_file.read(1)
        with rasterio.open(f"{sentinel2_raster_folder}/{raster_id}.tif") as sentinel2_file:
            sentinel2_raster = sentinel2_file.read()
            meta = sentinel2_file.meta.copy()
            count = sentinel2_file.count
            raster_height, raster_width = sentinel2_file.height, sentinel2_file.width
            meta.pop("nodata", None)
        if scale == "local":
            sentinel2_raster[count-1] = np.where(reference_label == 0, 0, sentinel2_raster[count-1])

        # add descriptions to the bands and save the final raster
        size_bytes = os.path.getsize(f"{sentinel2_raster_folder}/{raster_id}.tif")
        bigtiff_option = "YES" if size_bytes > 4 * 1024**3 else "NO"
        meta.update({"BIGTIFF":bigtiff_option})
        band_descriptions = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "CLD", "date_difference"]
        with rasterio.open(f"{sentinel2_raster_folder}/{raster_id}.tif", 'w', **meta, compress="LZW") as file:
            for band_index in range(count):
                file.write(sentinel2_raster[band_index], band_index + 1)
                file.set_band_description(band_index + 1, band_descriptions[band_index])
                file.nodata = None
        
        # remove the intermediary files
        for path in all_aoi_paths:
           os.remove(path)
        os.remove(f"{sentinel2_raster_folder}/{raster_id}.tif.aux.xml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download Sentinel 2 data for each of the AOIS")
    
    parser.add_argument("--data_folder", default=os.environ["DATA_FOLDER"], help="The path to the data folder.")
    parser.add_argument("--find_sentinel2_availability", action="store_true", default=False, help="Create metadata describing the availability of Sentinel 2 data.")
    parser.add_argument("--find_minimal_cloud_cover", action="store_true", default=False, help="From the available Sentinel 2 data, find the data with the minimum cloud cover.")
    parser.add_argument("--download_sentinel2", action="store_true", default=False, help="Download the Sentinel 2 data for each of the AOIs.")
    parser.add_argument("--create_sentinel2_rasters", action="store_true", default=False, help="Create raster files of the Sentinel 2 data, matching to the CEMS labels.")
    parser.add_argument("--scale", default="local", help="The scale at which to create raster files: local, nearby, context, con_context, or basin.")

    args = parser.parse_args()

    if args.find_sentinel2_availability:
        find_sentinel2_availability(args.data_folder, args.scale)

    if args.find_minimal_cloud_cover:
        find_minimal_cloud_cover(args.data_folder, args.scale)

    if args.download_sentinel2:
        download_sentinel2(args.data_folder, args.scale)
    
    if args.create_sentinel2_rasters:
        create_sentinel2_rasters(args.data_folder, args.scale)