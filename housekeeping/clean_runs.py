import argparse
from pathlib import Path
import subprocess
import yaml
import sys

def parse_args():

    p = argparse.ArgumentParser(description='Map sweep experiments to the GPUs they are running on')
    p.add_argument('output_dir', type=Path, help='Directory containing all the runs')
    p.add_argument('slurm_output_dir', type=Path, help='Directory containing slurm output files')
    p.add_argument('--dry', action='store_true', help='Dry run, do not delete files')
    p.add_argument('--step_cutoff', type=float, default=10, help='Epoch cutoff for stale runs')

    args = p.parse_args()

    # DELETE THIS LATER
    # print('develeopment setting: dry run is always on')
    # args.dry = True

    return args

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

def parse_slurm_file(slurm_file: Path):

    slurm_id = slurm_file_to_id(slurm_file)

    # open a file handler for slurm_file and get the first ten lines
    with open(slurm_file, 'r') as f:
        lines = f.readlines()

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


    return experiments


def is_stale_run(run_dir: Path, running_run_ids: list, step_cutoff: float) -> bool:
    
    run_id = run_dir.name.split('_')[-1]

    # check if this run is still running
    is_running = run_id in running_run_ids

    # if run_dir.name == 'geom-gaussian-b16_7imwds35':
    #     print('*****')
    #     print(is_running)
    #     print(run_id)
        
    #     # print all running_run_ids separate by \n
    #     print(*running_run_ids, sep='\n')

    #     print('*****')

    if is_running:
        return False

    # after this point, we are only interested in runs that are not running

    # iterate over checkpoints, recording the last epoch that was saved
    checkpoint_dir = run_dir / 'checkpoints'

    if not checkpoint_dir.exists():
        return True

    max_step = 0
    for ckpt_file in checkpoint_dir.iterdir():
        if ckpt_file.name == 'last.ckpt':
            continue
        
        epoch = ckpt_file.name.split('-')[0].split('=')[-1]
        current_step = int(ckpt_file.stem.split('_')[-1])
        max_step = max(max_step, current_step)

    # if run_dir.name == 'geom-gaussian_fatket3a':
    #     print('*****')
    #     print(max_epoch)
    #     print(epoch_cutoff)
    #     print(max_epoch < epoch_cutoff)
    #     print('*****')

    if max_step < step_cutoff:
        return True
    else:
        return False
    
if __name__ == "__main__":

    args = parse_args()

    # get username of the current user
    username = Path.home().name

    # get running job ids
    running_jobs = get_currently_running_jobs(username)

    # find all the runs in the directory
    # run_dirs = [ run_dir for run_dir in args.output_dir.iterdir() if run_dir.name != 'wandb' ]
    run_dirs = []
    date_dirs = []
    for date_dir in args.output_dir.iterdir():
        date_dirs.append(date_dir)
        for run_dir in date_dir.iterdir():
            run_dirs.append(run_dir)

    # get all files in args.slurm_output_dir that correspond to currently running jobs
    slurm_files = [ f for f in args.slurm_output_dir.iterdir() if f.is_file() ]
    slurm_files = [ f for f in slurm_files if slurm_file_to_id(f) in running_jobs ]

    # map all running jobs to experiments
    running_experiments = []
    for slurm_file in slurm_files:
        experiments = parse_slurm_file(slurm_file)
        running_experiments.extend(experiments)

    # get the wandb run ids of every running experiment
    run_ids = []
    for experiment in running_experiments:
        node_name, slurm_id, experiment_name, run_url = experiment
        run_id = run_url.split('/')[-1]
        run_ids.append(run_id)

    # check which runs are stale
    runs_to_delete = [ run_dir for run_dir in run_dirs if is_stale_run(run_dir, run_ids, step_cutoff=args.step_cutoff) ]

    # print out the files to delete
    print("runs to delete:")
    print(*runs_to_delete, sep='\n')

    if args.dry:
        sys.exit()

    for run_dir in runs_to_delete:
        print(f'deleting {run_dir}')
        subprocess.run(['rm', '-rf', run_dir])

    for date_dir in date_dirs:
        is_empty = not any(date_dir.iterdir())
        print(f'deleteing date dir: {date_dir}')
        if is_empty:
            subprocess.run(['rm', '-rf', date_dir])