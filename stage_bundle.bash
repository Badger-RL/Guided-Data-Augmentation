#!/usr/bin/sh

## TO USE ## 

# Install expect with homebrew if needed - brew install expect
# https://formulae.brew.sh/formula/expect

# Turn on globalprotect VPN to enable chtc access

# Make sure you can shh into both transfer.chtc.wisc.edu and submit1.chtc.wisc.edu 
# without getting asked for permissions other than password

# Replace all "USERNAME" with wisc cs username in all files
# and replace PASSWORD with wisc cs password in all files

# Expected file structure for repos/folders: (Can change in commands for ease of use)
# ./AbstractSim
# ./BHumanCodeRelease
# ./condor_utils

## START OF SCRIPT ##

USERNAME=$(python3 -c "import json; config_file = open('credentials.json','r'); config = json.load(config_file); print(config['username'])")
PASSWORD=$(python3 -c "import json; config_file = open('credentials.json','r'); config = json.load(config_file); print(config['password'])")



/usr/bin/expect -c "
spawn scp ./bundle.zip ./offlinerl.sub ./condor_execute.sh ${USERNAME}@submit1.chtc.wisc.edu:/home/${USERNAME}/
expect \"Password: \"
send \"${PASSWORD}\n\"
interact
"
