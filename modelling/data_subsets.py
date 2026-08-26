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

def create_overlap_events_subset(data_folder):

    ####### define the patches
    def convert_to_dataframe(patch_list, train_subset=True):
        if train_subset:
            subset = ["train"]*len(patch_list)
        else:
            subset = ["val_" + "_".join(patch.split("_")[:2]) for patch in patch_list]
        return pd.DataFrame({"patch": patch_list, 
                            "subset":subset,
                            "event":[patch.split("_")[0] for patch in patch_list], 
                            "subevent":["_".join(patch.split("_")[:2]) for patch in patch_list]})

    all_patches = os.listdir(f"{data_folder}/local/label")
    all_patches.sort()

    patches = pd.read_csv(f"{data_folder}/subsets/only_patches_data_subset.csv")
    split = pd.read_csv(f"{data_folder}/subsets/only_split_data_subset.csv")
    buffered_patches = pd.read_csv(f"{data_folder}/subsets/buffered_patches.csv")

    three_eval_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR659_2023-05-05", "EMSR692_2023-09-10", "EMSR416_2019-12-15"] if subevent in patch)], train_subset=False)
    all_eval_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR759_2024-09-21", "EMSR761_2024-09-19", "EMSR659_2023-05-05", "EMSR416_2019-12-15", 
                                                                                    "EMSR692_2023-09-10", "EMSR757_2024-09-18", "EMSR756_2024-09-18"] if subevent in patch)], train_subset=False)
    non_overlap_eval_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR759_2024-09-21", "EMSR759no_2023-09-29", "EMSR761_2024-09-19", "EMSR761no_2023-09-10",
                                                                                                                "EMSR757_2024-09-18", "EMSR757no_2023-09-07", "EMSR756_2024-09-18", "EMSR756no_2023-09-07"] if subevent in patch)])

    patches_three_eval = patches[patches["subevent"].isin(["EMSR659_2023-05-05", "EMSR692_2023-09-10", "EMSR416_2019-12-15"])]
    split_three_eval = split[split["subevent"].isin(["EMSR659_2023-05-05", "EMSR692_2023-09-10", "EMSR416_2019-12-15"])]
    buffered_patches_three_eval = buffered_patches[buffered_patches["subevent"].isin(["EMSR659_2023-05-05", "EMSR692_2023-09-10", "EMSR416_2019-12-15"])]

    good_overlap_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR712_2023-12-25", "EMSR712_2024-01-03", "EMSR712_2024-01-06", 
    "EMSR261_2017-12-17", "EMSR261_2017-07-29", "EMSR142_2015-10-17", "EMSR142_2015-10-21",
    "EMSR768_2024-10-04", "EMSR768_2024-10-08", "EMSR698_2023-10-08", "EMSR783_2025-01-01", "EMSR273_2018-03-12", "EMSR273_2018-03-14", "EMSR273_2018-03-22", "EMSR501_2021-02-12",
    "EMSR501_2021-02-15", "EMSR555_2021-12-14", "EMSR555_2021-12-11", "EMSR555_2021-12-10", "EMSR279_2018-04-13", "EMSR279_2018-04-14", "EMSR659no_2022-05-18", "EMSR762_2024-09-20", 
    "EMSR771_2024-10-25", "EMSR664_2023-05-18", "EMSR429_2020-02-26", "EMSR149_2016-01-10", "EMSR548_2021-11-02", "EMSR649_2023-02-11", "EMSR492_2021-01-02", "EMSR416no_2019-10-11", 
    "EMSR437_2020-05-13", "EMSR692no_2022-09-30", "EMSR465_2020-09-20", "EMSR271_2018-03-01"] if subevent in patch)])

    all_overlap_events = convert_to_dataframe([patch for patch in all_patches if any(True for event in ["EMSR271", "EMSR465", "EMSR492", "EMSR437", "EMSR762", "EMSR771", 
                        "EMSR548", "EMSR649", "EMSR142", "EMSR768", "EMSR261", "EMSR712", "EMSR273", "EMSR501", "EMSR555",
                        "EMSR279", "EMSR429", "EMSR149", "EMSR698", "EMSR783", "EMSR664", "EMSR692no", "EMSR416no", "EMSR659no"] if event in patch)])

    all_overlap_greece_events = convert_to_dataframe([patch for patch in all_patches if any(True for event in ["EMSR271", "EMSR465", "EMSR692no"] if event in patch)])
    good_overlap_greece_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR271_2018-03-01", "EMSR465_2020-09-20", "EMSR692no_2022-09-30"] if subevent in patch)])
    greece_split = split[split["subevent"]=="EMSR692_2023-09-10"]

    all_overlap_france_events = convert_to_dataframe([patch for patch in all_patches if any(True for event in ["EMSR492", "EMSR437", "EMSR416no"] if event in patch)])
    good_overlap_france_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR492_2021-01-02", "EMSR437_2020-05-13", "EMSR416no_2019-10-11"] if subevent in patch)])
    france_split = split[split["subevent"]=="EMSR416_2019-12-15"]

    all_overlap_italy_events = convert_to_dataframe([patch for patch in all_patches if any(True for event in ["EMSR762", "EMSR771", "EMSR659no"] if event in patch)])
    good_overlap_italy_subevents = convert_to_dataframe([patch for patch in all_patches if any(True for subevent in ["EMSR762_2024-09-20", "EMSR659no_2022-05-18", "EMSR771_2024-10-25"] if subevent in patch)])
    italy_split = split[split["subevent"]=="EMSR659_2023-05-05"]

    europe_events = pd.read_csv(f"{data_folder}/metadata/subevent_descriptions.csv")
    europe_events = europe_events[(europe_events["continent"]=="Europe") & (europe_events["flood_cause"] != "snow_melt")]
    included_events = list(set(europe_events["event"])-{"EMSR759", "EMSR761", "EMSR659", "EMSR416", "EMSR692", "EMSR757", "EMSR756"}) + ["EMSR273", "EMSR279", "EMSR416no", "EMSR659no", "EMSR692no", "EMSR759no", "EMSR761no", "EMSR757no", "EMSR756no"]
    europe_patches = convert_to_dataframe([patch for patch in all_patches if any(event in patch for event in included_events)])

    ####### define the training subsets

    # predict the full eval subevent after training on repeated events
    # (can a new event be predicted after training on previous events)
    all_overlap = pd.concat([all_overlap_events, all_eval_subevents], ignore_index=True) 
    good_overlap = pd.concat([good_overlap_subevents, all_eval_subevents], ignore_index=True)
    all_overlap_three = pd.concat([all_overlap_events, non_overlap_eval_subevents, three_eval_subevents], ignore_index=True) 
    good_overlap_three = pd.concat([good_overlap_subevents, non_overlap_eval_subevents, three_eval_subevents], ignore_index=True)

    # predict the split/patches/buffered patches of the three eval subevents after training on repeated events and the train patches from the three eval subevents
    # (does training on previous events improve performance of training on split/patches/buffered patches of a new event)
    all_overlap_split = pd.concat([all_overlap_events, non_overlap_eval_subevents, split_three_eval], ignore_index=True)
    good_overlap_split = pd.concat([good_overlap_subevents, non_overlap_eval_subevents, split_three_eval], ignore_index=True)
    all_overlap_patches = pd.concat([all_overlap_events, non_overlap_eval_subevents, patches_three_eval], ignore_index=True)
    good_overlap_patches = pd.concat([good_overlap_subevents, non_overlap_eval_subevents, patches_three_eval], ignore_index=True)
    all_buffered_overlap_patches = pd.concat([all_overlap_events, non_overlap_eval_subevents, buffered_patches_three_eval], ignore_index=True)
    good_buffered_overlap_patches = pd.concat([good_overlap_subevents, non_overlap_eval_subevents, buffered_patches_three_eval], ignore_index=True)

    # predict a full subevent after training only on its previous events
    # (can a new flood event be predicted after training on a few previous events in only one location)
    all_overlap_greece = pd.concat([all_overlap_greece_events, three_eval_subevents[three_eval_subevents["event"]=="EMSR692"]], ignore_index=True)
    all_overlap_france = pd.concat([all_overlap_france_events, three_eval_subevents[three_eval_subevents["event"]=="EMSR416"]], ignore_index=True)
    all_overlap_italy = pd.concat([all_overlap_italy_events, three_eval_subevents[three_eval_subevents["event"]=="EMSR659"]], ignore_index=True)
    good_overlap_greece = pd.concat([good_overlap_greece_subevents, three_eval_subevents[three_eval_subevents["event"]=="EMSR692"]], ignore_index=True)
    good_overlap_france = pd.concat([good_overlap_france_subevents, three_eval_subevents[three_eval_subevents["event"]=="EMSR416"]], ignore_index=True)
    good_overlap_italy = pd.concat([good_overlap_italy_subevents, three_eval_subevents[three_eval_subevents["event"]=="EMSR659"]], ignore_index=True)

    # predict a subevent split after training only on its previous events and the other split half
    # (does training on a few previous events in just that location help with a split prediction)
    all_overlap_greece_split = pd.concat([all_overlap_greece_events, greece_split], ignore_index=True)
    all_overlap_france_split = pd.concat([all_overlap_france_events, france_split], ignore_index=True)
    all_overlap_italy_split = pd.concat([all_overlap_italy_events, italy_split], ignore_index=True)
    good_overlap_greece_split = pd.concat([good_overlap_greece_subevents, greece_split], ignore_index=True)
    good_overlap_france_split = pd.concat([good_overlap_france_subevents, france_split], ignore_index=True)
    good_overlap_italy_split = pd.concat([good_overlap_italy_subevents, italy_split], ignore_index=True)

    # predict the split/patches/buffered patches of all the eval subevents after training on repeated events and the train patches from all the eval subevents
    # (does training on repeated overlap events help predictions even for events that are not within the overlap)
    all_overlap_split_all = pd.concat([all_overlap_events, split], ignore_index=True)
    good_overlap_split_all = pd.concat([good_overlap_subevents, split], ignore_index=True)
    all_overlap_patches_all = pd.concat([all_overlap_events, patches], ignore_index=True)
    good_overlap_patches_all = pd.concat([good_overlap_subevents, patches], ignore_index=True)
    all_buffered_overlap_patches_all = pd.concat([all_overlap_events, buffered_patches], ignore_index=True)
    good_buffered_overlap_patches_all = pd.concat([good_overlap_subevents, buffered_patches], ignore_index=True)

    # predict the full eval subevents after training on data from all europe, including repeated and non overlap events
    # (can also be used as a pretrained model to finetune from)
    europe_subset = pd.concat([europe_patches, all_eval_subevents], ignore_index=True) 

    # save the csv subset files
    subset_csvs = [all_overlap, good_overlap, all_overlap_three, good_overlap_three, all_overlap_split, good_overlap_split, all_overlap_patches, good_overlap_patches, all_buffered_overlap_patches,
    good_buffered_overlap_patches, all_overlap_greece, all_overlap_france, all_overlap_italy, good_overlap_greece, good_overlap_france, good_overlap_italy, all_overlap_greece_split,
    all_overlap_france_split, all_overlap_italy_split, good_overlap_greece_split, good_overlap_france_split, good_overlap_italy_split, all_overlap_split_all, good_overlap_split_all,
    all_overlap_patches_all, good_overlap_patches_all, all_buffered_overlap_patches_all, good_buffered_overlap_patches_all, europe_subset]
    subset_names = ["all_overlap.csv", "good_overlap.csv", "all_overlap_three.csv", "good_overlap_three.csv", "all_overlap_split.csv", "good_overlap_split.csv", "all_overlap_patches.csv",
    "good_overlap_patches.csv", "all_buffered_overlap_patches.csv", "good_buffered_overlap_patches.csv", "all_overlap_greece.csv", "all_overlap_france.csv", "all_overlap_italy.csv",
    "good_overlap_greece.csv", "good_overlap_france.csv", "good_overlap_italy.csv", "all_overlap_greece_split.csv", "all_overlap_france_split.csv", "all_overlap_italy_split.csv",
    "good_overlap_greece_split.csv", "good_overlap_france_split.csv", "good_overlap_italy_split.csv", "all_overlap_split_all.csv", "good_overlap_split_all.csv",
    "all_overlap_patches_all.csv", "good_overlap_patches_all.csv", "all_buffered_overlap_patches_all.csv", "good_buffered_overlap_patches_all.csv", "europe_subset.csv"]
    for subset_csv, subset_name in zip(subset_csvs, subset_names):
        subset_csv.to_csv(f"{data_folder}/subsets/{subset_name}", index=False)

    # create datasubsets for the subevents training 
    train_patches = pd.read_csv(f"{data_folder}/subsets/only_patches_data_subset.csv")
    train_patches["subset"] = "train"
    subevent_788 = pd.read_csv(f"{data_folder}/subsets/del_subset_emsr788.csv")
    subevent_788 = subevent_788.replace("val", "val_788")
    subevent_763 = pd.read_csv(f"{data_folder}/subsets/del_event_test_data_subset.csv")
    subevent_763 = subevent_763.replace("val", "val_763")
    both_subevents = pd.concat([subevent_788, subevent_763], ignore_index=True) 
    subevents_with_eval = pd.concat([both_subevents, train_patches], ignore_index=True) 

    subset_csvs = [subevent_788, subevent_763, both_subevents, subevents_with_eval]
    subset_names = ["subevent_788.csv", "subevent_763.csv", "both_subevents.csv", "subevents_with_eval.csv"]
    for subset_csv, subset_name in zip(subset_csvs, subset_names):
        subset_csv.to_csv(f"{data_folder}/subsets/{subset_name}", index=False)