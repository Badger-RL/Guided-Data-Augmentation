#!/usr/bin/sh

# Removes all outputs for staging and transfer connections
# Transfers results into a folder ./results (which needs to be made BEFORE)

USERNAME=$(python3 -c "import json; config_file = open('credentials.json','r'); config = json.load(config_file); print(config['username'])")
PASSWORD=$(python3 -c "import json; config_file = open('credentials.json','r'); config = json.load(config_file); print(config['password'])")

echo "$user"


/usr/bin/expect -c "
spawn ssh ${USERNAME}@submit1.chtc.wisc.edu \"rm -rf ./result_*\"
expect \"Password: \"
send \"${PASSWORD}\n\"
interact
"

