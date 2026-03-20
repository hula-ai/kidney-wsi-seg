import os
import json


path = "/data/syed/filtered_TMA_4096_tiles.json"

with open(path, "r") as f:
    filtered_tile_names = json.load(f)

# print(filtered_tile_names.keys())
case_name_dict = {}
for key, name_list in filtered_tile_names.items():
    case_name_dict[key] = []
    for tile_fname in name_list:
        fname = " ".join(tile_fname.split(" ")[0:2]) if "WCM" not in tile_fname \
            else tile_fname.split(" [x=")[0]
        if fname not in case_name_dict[key]:
            case_name_dict[key].append(fname)

for key, item in case_name_dict.items():
    print("Fold {}:".format(key), item)

with open("./case_names.json", "w", encoding='utf-8') as f:
    json.dump(case_name_dict, f, ensure_ascii=False, indent=4)

