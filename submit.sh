mkdir -p $3
condor_submit offlinerl.sub num_jobs=$2 commands_file=src/condor/commands/${1} output_dir=$3
