import json
import itertools


TRAINING_RUN_LENGTH = 100000
EVAL_FREQ = 5000
COUNT = 5




def add_settings(manifest, hyperparams, dataset_name, dataset_path):
    keys, values =zip(*hyperparams.items())
    for value in itertools.product(*values):
        entry = dict(zip(keys,value))
        entry["name"] = dataset_name + "_" + "_".join([str(k) + "_" + str(v) for k,v in entry.items()])
        entry["dataset_name"] = dataset_path
        entry["eval_freq"] = EVAL_FREQ
        entry["training_steps"] = TRAINING_RUN_LENGTH
        entry["count"] = COUNT
        manifest.append(entry)





if __name__ == "__main__":

    hyperparams={"qf_lr": [5e-6, 3e-6, 1e-4],
                "soft_target_update_rate":[0.005, 0.05]}


    manifest = []
    add_settings(manifest, hyperparams, "expert_augmented_100000", "expert/translate_robot_and_ball/100000_1.hdf5")
    add_settings(manifest, hyperparams, "expert_100000", "expert/100000.hdf5")
    result = {}
    result["total_jobs"] = sum([entry["count"] for entry in manifest])
    result["experiment_manifest"] = manifest
    with open("sweep_config.json", 'w') as config_file:
        json.dump(result, config_file, indent=4)
