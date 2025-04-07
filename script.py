import os
import geopandas as gpd

all_aoi = [os.path.join(root, file) for root, dirs, files in os.walk("data") for file in files if file.endswith(".json") and "areaOfInterestA" in file]
all_obs = [os.path.join(root, file) for root, dirs, files in os.walk("data") for file in files if file.endswith(".json") and "observedEventA" in file]

print(all_aoi)
print(all_obs)