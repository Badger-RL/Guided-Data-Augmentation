import wandb
import json
import numpy as np
from pathlib import Path

def filter_entries(key_filter, entry):
    result = None
    for key in key_filter:
        if key in entry:
            if result == None:
                result = {"timestamp": entry["_timestamp"]}
            result[key] = entry[key]
    return result

wandb.login(key = "PLACEHOLDER")

api = wandb.Api()

runs = api.runs(path="balis/CORL")



key_filter =[ "d4rl_normalized_score", "success_rate"]


def get_name_prefix(name):
    token_list = name.split("_")
    return "_".join(token_list[:-1])
    
"""
for run in runs:
    #print(run.name)
    #metrics = list(run.scan_history())
    print(get_name_prefix(run.name))
    save_folder = f"./src/logdata/{get_name_prefix(run.name)}/"
    Path(save_folder).mkdir(exist_ok = True, parents = True)
    continue
    result[run.name] = []
    for entry in metrics:
            entry_result = filter_entries(key_filter, entry)
            if entry_result != None:
                result[run.name].append(entry_result)
    print(result[run.name])
"""

for run in runs:
    #print(run.name)
    metrics = run.history(keys= key_filter)
    print(get_name_prefix(run.name))
    save_folder = f"./src/logdata/{get_name_prefix(run.name)}/"
    Path(save_folder).mkdir(exist_ok = True, parents = True)
    result = {"t":[],"r":[],"reward":[]}
    print(metrics)
    for i,row in metrics.iterrows():
            entry_result = row# filter_entries(key_filter, entry)
            print(entry_result)
            result["t"].append(entry_result["_step"])
            result["reward"].append(entry_result["d4rl_normalized_score"])
            result["r"].append(entry_result["success_rate"])
    print(result)
    np.savez(f"{save_folder}{run.name}{'.npz'}", **result)


#with open("log.json", 'w') as json_file:
#    json.dump(result, json_file)

#data = {}
#with open("log.json", 'r') as json_file:
#    data = json.load(json_file)
#for key in data:
#    result = {"t":[],"r":[]}
#    for entry in data[key]:
#        result["t"].append(entry["timestamp"])
#        result["r"].append(entry["d4rl_normalized_score"])
#    with open(f"./logdata/{key}.json","wb") as out_file:
#        np.save(out_file, result)
#with open("./src/logdata/random_augmented_100000/random_augmented_100000_113.npz", 'rb') as saved_file:
#    print(dict(np.load(saved_file)))