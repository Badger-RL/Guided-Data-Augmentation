# $1 = name of commands file
# $2 = number of seeds per command
# $3 = where results be copied to on the submit node.
mkdir -p results/$3
condor_submit condor_execute.sub num_jobs=$2 commands_file=commands/${1} output_dir=results/$3
