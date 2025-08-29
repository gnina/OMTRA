import argparse
from pathlib import Path
import subprocess

def parse_args():

    p = argparse.ArgumentParser(description='Remove all slurm output except currently running jobs')
    p.add_argument('slurm_output_dir', type=Path, help='Directory containing slurm output files')
    p.add_argument('--dry', action='store_true', help='Dry run, do not delete files')

    return p.parse_args()

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

if __name__ == "__main__":

    args = parse_args()

    # get username of the current user
    username = Path.home().name

    # get running job ids
    running_jobs = get_currently_running_jobs(username)

    # get all files in args.slurm_output_dir
    slurm_files = [ f for f in args.slurm_output_dir.iterdir() if f.is_file() ]

    # iterate over all slurm files
    files_to_delete = []
    for slurm_file in slurm_files:
        if slurm_file.stem in running_jobs:
            continue
        files_to_delete.append(slurm_file)

    # print out the files to delete
    print("Files to delete:")
    print(*files_to_delete, sep='\n')

    if not args.dry:
        for file in files_to_delete:
            file.unlink()