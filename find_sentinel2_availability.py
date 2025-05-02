import pandas as pd
import geopandas as gpd
import os
import pandas as pd
import time
from creds import *
from tqdm import tqdm
from sentinelhub import (SHConfig, DataCollection, SentinelHubCatalog, CRS, Geometry)

# configure the API credentials
config = SHConfig()
config.sh_client_id = os.environ["COPERNICUS_CLIENT_ID"]
config.sh_client_secret = os.environ["COPERNICUS_CLIENT_SECRET"]
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.save("cdse")
config = SHConfig("cdse")
catalog = SentinelHubCatalog(config=config)

# import the aois that we want to find Sentinel 2 data for
aois = gpd.read_file("metadata/aoi_extent.geojson")
aois = aois.drop_duplicates(["geometry_event_date_id"], ignore_index=True)

# get metadata on available sentinel 2 data with its id, time, and cloud cover for the given event and date
availability = []
for index in tqdm(range(len(aois))):

    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        geometry=Geometry(aois.loc[index, "geometry"], CRS.WGS84),
        time=(aois.loc[index, "earlier_date"], aois.loc[index, "event_date"]),
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []})
    
    results = pd.DataFrame([{"id": item["id"], 
                             "datetime": item["properties"]["datetime"], 
                             "cloud_cover": item["properties"]["eo:cloud_cover"]} for item in list(search_iterator)])
    
    for attribute in ["event", "event_date", "geometry_event_date_id"]:
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
availability.to_csv("metadata/sentinel2_availability.csv", index=False) 