import argparse
from pathlib import Path
import subprocess

def parse_args():

    p = argparse.ArgumentParser(description='Map sweep experiments to the GPUs they are running on')
    p.add_argument('slurm_output_dir', type=Path, help='Directory containing slurm output files')
    p.add_argument('--running', action='store_true', help='Only print experiments that are currently running')

    return p.parse_args()

def parse_slurm_file(slurm_file: Path, running: bool = False):

    # open a file handler for slurm_file and get the first ten lines
    with open(slurm_file, 'r') as f:
        lines = f.readlines()

    slurm_id = slurm_file.name.split('.')[0].split('-')[-1]

    # # get the line that contains the node hostname
    # node_line = [ line for line in lines[:10] if '.csb.pitt.edu' in line ]

    # if len(node_line) == 0:
    #     print(f'unable to parse {slurm_file}', flush=True)
    #     return None
    
    # node_line = node_line[0]
    node_line = lines[0]
    node_name = node_line.strip()

    # iterate over all remaining lines in file and get the line number of the line that contains 'View run at'
    view_run_line_idxs = []
    for idx, line in enumerate(lines):
        if 'View run at' in line:
            view_run_line_idxs.append(idx)


    experiments = []
    for view_run_line_idx in view_run_line_idxs:
        run_url = lines[view_run_line_idx].split(' ')[-1].strip()
        # print(run_url)
        experiment_name = lines[view_run_line_idx - 2].split(' ')[-1].strip()
        experiments.append((node_name, slurm_id, experiment_name, run_url))

    if len(experiments) == 0:
        return None
    if running:
        return [ experiments[-1] ]

    return experiments

def get_currently_running_jobs(username: str):

    # run the command line command "squeue -u <username>" and get the output
    squeue_output = subprocess.check_output(['squeue', '-u', username]).decode('utf-8')

    squeue_output = squeue_output.split('\n')

    process_ids = []

    for job_line in squeue_output[1:]:

        job_line = job_line.split(' ')
        job_line = [ x for x in job_line if x != '' ]
        if len(job_line) == 0:
            continue
        process_id = job_line[0]
        process_ids.append(process_id)


    return set(process_ids)

def slurm_file_to_id(slurm_file: Path):
    return slurm_file.name.split('.')[0]

if __name__ == "__main__":

    args = parse_args()


    if args.running:
        # get username of the current user
        username = Path.home().name

        # get running job ids
        running_jobs = get_currently_running_jobs(username)
    else:
        running_jobs = []

    # get all files in args.slurm_output_dir
    slurm_files = [ f for f in args.slurm_output_dir.iterdir() if f.is_file() ]
    if args.running:
        slurm_files = [ f for f in slurm_files if slurm_file_to_id(f) in running_jobs]

    # iterate over all slurm files
    for slurm_file in slurm_files:
        experiments = parse_slurm_file(slurm_file, running=args.running)
        if experiments is None:
            continue

        for experiment in experiments:
            print('\t'.join(experiment), flush=True)


