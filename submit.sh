mkdir -p results/$3
condor_submit condor_execute.sub num_jobs=$2 commands_file=commands/${1} output_dir=results/$3
