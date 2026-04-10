import pandas as pd
import os
import json
import geopandas as gpd

def create_patches_checkerboard(data_folder, all_patches, subevent):
    
    patch_drop = json.load(open(f"{data_folder}/metadata/subevent_patches.json"))[subevent]["patches_per_column"]
    subevent_patches = [patch for patch in all_patches if subevent in patch]
    subset = []
    for patch in subevent_patches:
        patch_num = int(patch.split("_")[-1][:-4])
        row = patch_num // patch_drop
        col = patch_num % patch_drop
        subset.append("train" if (row + col) % 2 == 0 else f"val_{subevent}")
    return pd.DataFrame({"patch":subevent_patches, "subset":subset})

def create_data_subset(data_folder):
        
    # only include patches from europe, that were not caused by snow melt
    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()
    events = pd.read_csv(f"{data_folder}/metadata/subevent_descriptions.csv")
    events = events[events["continent"]=="Europe"]
    events = events[events["flood_cause"] != "snow_melt"]
    included_events = set(events["event"])
    all_patches = [patch for patch in all_patches if any(subevent in patch for subevent in included_events)]

    # create train and val patches in a checkerboard for the provided subevents
    emsr756_patches = list(gpd.read_file(f"{data_folder}/metadata/emsr756_geometry.geojson")["patch"])
    emsr756_patches = [patch for patch in all_patches if patch in emsr756_patches]
    patches_subset = create_patches_checkerboard(data_folder, emsr756_patches, "EMSR756_2024-09-18")
    subevents = ["EMSR759_2024-09-21", "EMSR761_2024-09-19", "EMSR416_2019-12-15", "EMSR659_2023-05-05", "EMSR692_2023-09-10", "EMSR757_2024-09-18"]
    for subevent in subevents:
        new_patches_subset = create_patches_checkerboard(data_folder, all_patches, subevent)
        patches_subset = pd.concat([patches_subset, new_patches_subset])

    # create val patches for the full subevents
    for subevent in ["EMSR788_2025-02-04", "EMSR763_2024-10-07"]:
        data = [patch for patch in all_patches if subevent in patch]
        patches_subset = pd.concat([patches_subset, pd.DataFrame({"patch":data, "subset":[f"val_{subevent}"]*len(data)})])

    # remove the other subevents from the same event that we evaluate on
    excluded = ["EMSR756", "EMSR759", "EMSR761", "EMSR416", "EMSR659", "EMSR692", "EMSR757", "EMSR788_2025-02-05", "EMSR788_2025-02-04", "EMSR763_2024-10-07"]
    all_patches = [patch for patch in all_patches if not any(subevent in patch for subevent in excluded)]

    patches_subset = pd.concat([patches_subset, pd.DataFrame({"patch":all_patches, "subset": ["train"]*len(all_patches)})])
    patches_subset["event"] = patches_subset["patch"].str.split("_").str[0]
    patches_subset["subevent"] = patches_subset["patch"].str.split("_").str[0] + "_" + patches_subset["patch"].str.split("_").str[1]
    patches_subset.to_csv(f"{data_folder}/subsets/europe_data_subset.csv", index=False)

def create_eval_subset(data_folder):

    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()

    emsr756_patches = list(gpd.read_file(f"{data_folder}/metadata/emsr756_geometry.geojson")["patch"])
    emsr756_patches = [patch for patch in all_patches if patch in emsr756_patches]
    patches_subset = create_patches_checkerboard(data_folder, emsr756_patches, "EMSR756_2024-09-18")
    subevents = ["EMSR759_2024-09-21", "EMSR761_2024-09-19", "EMSR416_2019-12-15", "EMSR659_2023-05-05", "EMSR692_2023-09-10", "EMSR757_2024-09-18"]
    for subevent in subevents:
        new_patches_subset = create_patches_checkerboard(data_folder, all_patches, subevent)
        patches_subset = pd.concat([patches_subset, new_patches_subset])

    patches_subset["event"] = patches_subset["patch"].str.split("_").str[0]
    patches_subset["subevent"] = patches_subset["patch"].str.split("_").str[0] + "_" + patches_subset["patch"].str.split("_").str[1]
    patches_subset.to_csv(f"{data_folder}/subsets/only_patches_data_subset.csv", index=False)
    
def create_subevent_subset(data_folder, subevent):

    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()
    event = subevent.split("_")[0]
    val_patches = [patch for patch in all_patches if any(subevent in patch for subevent in [subevent])]
    train_patches = [patch for patch in all_patches if event in patch]
    train_patches = list(set(train_patches)-set(val_patches))
    train_patches = [patch for patch in train_patches if not ("EMSR788_2025-02-05" in patch)]
    subset_df = pd.concat([pd.DataFrame({"patch":val_patches, "subset":["val"]*len(val_patches)}),
                            pd.DataFrame({"patch":train_patches, "subset":["train"]*len(train_patches)})])
    subset_df["event"] = subset_df["patch"].str.split("_").str[0]
    subset_df["subevent"] = subset_df["patch"].str.split("_").str[0] + "_" + subset_df["patch"].str.split("_").str[1]
    subset_df.to_csv(f"{data_folder}/subsets/subevent_{event}_subset.csv", index=False)