import geopandas as gpd
import os
import argparse
from creds import *
from tqdm import tqdm
import time
import math
from sentinelhub import (SHConfig, DataCollection, SentinelHubRequest, BBox, BBoxSplitter, bbox_to_dimensions, CRS, MimeType, Geometry, MosaickingOrder)

def main(save_folder):
    # configure the API credentials
    config = SHConfig()
    config.sh_client_id = os.environ["COPERNICUS_CLIENT_ID"]
    config.sh_client_secret = os.environ["COPERNICUS_CLIENT_SECRET"]
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")
    config = SHConfig("cdse")

    # import the aoi data with the Sentinel 2 availability
    aois = gpd.read_file("metadata/aoi_availability.geojson")
    aois = aois.drop_duplicates(["geometry_event_date_id"], ignore_index=True)
    aois["availability_date"] = aois["availability_date"].dt.date

    for index in tqdm(range(len(aois))):

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

        # extract the sentinel 2 bands and save as a GeoTiff in UINT16
        for bbox in tqdm(split_aois, leave=False):
            bbox_size = bbox_to_dimensions(bbox, resolution=10)
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
                        time_interval=(aois.loc[index, "availability_date"], aois.loc[index, "availability_date"]),
                        mosaicking_order=MosaickingOrder.MOST_RECENT
                        )],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=bbox_size,
                geometry=geometry,
                config=config,
                data_folder=f"{save_folder}/aoi_{aois.loc[index, 'geometry_event_date_id']}"
                )
            
            data = request.get_data(save_data=True,
                                    redownload=False)
            
            time.sleep(0.5)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download Sentinel 2 data for each of the AOIS")
    parser.add_argument("--save_folder", required=True, help="The folder within which to save the Sentinel 2 data")
    args = parser.parse_args()

    main(save_folder=args.save_folder)