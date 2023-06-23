#!/usr/bin/bash

# Removes all outputs for staging and transfer connections
# Transfers results into a folder ./results (which needs to be made BEFORE)


USERNAME=$(python3 -c "import json; config_file = open('credentials.json','r'); config = json.load(config_file); print(config['username'])")
PASSWORD=$(python3 -c "import json; config_file = open('credentials.json','r'); config = json.load(config_file); print(config['password'])")



if [ $# -ne "1" ]; then
    echo "usage: get_results.sh <params_file>
       get_results.sh <save_dir_name>"
    exit 1
fi


OUTPUT_PATH = "./src/logdata"



/usr/bin/expect -c "
spawn ssh ${USERNAME}@submit1.chtc.wisc.edu \"rm results.zip\"
expect \"Password: \"
send \"${PASSWORD}\n\"
interact
"
/usr/bin/expect -c "
spawn ssh ${USERNAME}@submit1.chtc.wisc.edu \"cat ./result_*/*.tar.gz | tar -xzvf - -i\"
expect \"Password: \"
send \"${PASSWORD}\n\"
interact
"

/usr/bin/expect -c "
spawn ssh ${USERNAME}@submit1.chtc.wisc.edu \"zip -r results.zip ./results\"
expect \"Password: \"
send \"${PASSWORD}\n\"
interact
"




/usr/bin/expect -c "
spawn scp -r ${USERNAME}@submit1.chtc.wisc.edu:/home/${USERNAME}/results.zip ./src/logdata/$1
expect \"Password: \"
send \"${PASSWORD}\n\"
interact
"