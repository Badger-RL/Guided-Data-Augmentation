import wandb
import json
import numpy as np
import os
from pathlib import Path
import sys

def filter_entries(key_filter, entry):
    result = None
    for key in key_filter:
        if key in entry:
            if result == None:
                result = {"timestamp": entry["_timestamp"]}
            result[key] = entry[key]
    return result


if __name__ == "__main__":





    if len(sys.argv) != 2:
        print("usage: python3 ./pull_results.py <batch_name>")


    wandb_key = None
    with open("wandb_credentials.json", 'r') as credentials_file:
        credentials_json = json.load(credentials_file)
        wandb_key = credentials_json['wandb_key']





    wandb.login(key = wandb_key)

    api = wandb.Api()

    runs = api.runs(path="balis/CORL")



    key_filter =[ "d4rl_normalized_score", "success_rate", "average_target_q", "policy_loss",
    "qf1_loss", "qf2_loss", "alpha_loss", "log_pi", "cql_std_q1", "cql_std_q2"]



  

    for run in runs:
        metrics = run.history(keys= key_filter)
        print(run.name)
        if run.state != "finished":
            continue
        #if not "100k" in run.config["dataset_name"]:
        #    continue

        save_folder = f"./src/logdata/{run.name.replace('/', '_')}/"  
        Path(save_folder).mkdir(exist_ok = True, parents = True)
        result = {"t":[],"r":[],"success_rate":[], "average_target_q" : [], "policy_loss": [], 
        "qf1_loss" : [], "qf2_loss":[], "alpha_loss" : [], "log_pi": [], "cql_std_q1":[],
        "cql_std_q2":[]}
        print(metrics)
        for i,row in metrics.iterrows():
                entry_result = row# filter_entries(key_filter, entry)
                print(entry_result)
                result["t"].append(entry_result["_step"])
                result["r"].append(entry_result["d4rl_normalized_score"])
                result["average_target_q"].append(entry_result["average_target_q"])
                result["policy_loss"].append(entry_result["policy_loss"])
                result["qf1_loss"].append(entry_result["qf1_loss"])
                result["qf2_loss"].append(entry_result["qf2_loss"])
                result["alpha_loss"].append(entry_result["alpha_loss"])
                result["log_pi"].append(entry_result["log_pi"])
                result["cql_std_q1"].append(entry_result["cql_std_q1"])
                result["cql_std_q2"].append(entry_result["cql_std_q2"])
                result["success_rate"].append(entry_result["success_rate"])
        print(result)
        index = 0
        while os.path.isfile(f"{save_folder}{'run'}_{str(index)}{'.npz'}"):
            index+=1
        np.savez(f"{save_folder}{'run'}_{str(index)}{'.npz'}", **result)


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