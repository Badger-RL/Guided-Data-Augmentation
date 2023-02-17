import json
import sys
import subprocess



def check_config_integrity(config):
    assert (sum([experiment["count"] for experiment in config["experiment_manifest"] ]) == config["total_jobs"])
    for experiment in config["experiment_manifest"]:
        assert(experiment["count"] > 0)

def get_manifest_index(config, job_index):
    total = 0
    for i in range(len(config["experiment_manifest"])):
        total = total + config["experiment_manifest"][i]["count"]
        if total > job_index:
            return i
    
    
                                                           



if __name__ == "__main__":

    config = None
    with open("../config.json", 'r') as json_file:
        config = json.load(json_file)


    check_config_integrity(config)

    index = int(sys.argv[1]) # index is 0 through n-1 where n is the number of jobs
    manifest_index = get_manifest_index(config, index)

    command_list = ["python3", "./algorithms/cql.py"]
    # job name
    command_list.append("--name")
    command_list.append( f"{config['experiment_manifest'][manifest_index]['name']}_{str(index)}" )
    #dataset name
    command_list.append("--dataset_name")
    command_list.append(config["experiment_manifest"][manifest_index]["dataset_name"])
    # number of training steps 
    command_list.append("--max_timesteps")
    command_list.append(str(config["experiment_manifest"][manifest_index]["training_steps"]))
    # checkpoint eval frequency
    command_list.append("--eval_freq")
    command_list.append(str(config["experiment_manifest"][manifest_index]["eval_freq"]))
    #sets the seed to the job index (note that this means not controlling for seeds across experimental cases right now)
    command_list.append("--seed")
    command_list.append(str(index))






    subprocess.run( command_list)

