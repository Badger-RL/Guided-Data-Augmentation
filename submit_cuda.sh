output_dir=results/$3
mkdir -p $output_dir
condor_submit condor_execute_cuda.sub num_jobs=$2 commands_file=commands/${1} output_dir=$output_dir
