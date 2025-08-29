import os
from pathlib import Path
from pipeline_components import SystemProcessor
from crossdocked_unlink_zarr import CrossdockedNoLinksZarrConverter
from rdkit import Chem
import logging
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time
import traceback
#logging.basicConfig(level=logging.DEBUG)
logging.disable(level=logging.CRITICAL)  # Set to CRITICAL to suppress debug output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.vlen_utf8")
import torch

# Crossdocked Data Directories

types_file_dir ="/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types"
root_dir = "/net/galaxy/home/koes/paf46_shared/cd2020_v1.3"
train_zarr_output_dir = "train_external_output.zarr"
test_zarr_output_dir = "test_external_output.zarr"

#load the file
data = torch.load('crossdocked_external_splits/split_by_name.pt')

def parse_args():
    parser = argparse.ArgumentParser(description="Test Crossdocked dataset processing")
    parser.add_argument("--pocket_cutoff", type=float, default=8.0, help="Pocket cutoff distance")
    parser.add_argument("--train_zarr_output_dir", type=str, default=train_zarr_output_dir, help="Output Zarr directory for training")
    parser.add_argument("--test_zarr_output_dir", type=str, default=test_zarr_output_dir, help="Output Zarr directory for testing")
    parser.add_argument("--root_dir", type=str, default=root_dir, help="Root directory for crossdocked data")

    parser.add_argument("--max_batches", type=str, default="None", help="Maximum number of batches to process")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for processing ligand-receptor pairs")
    parser.add_argument("--n_cpus", type=int, default=8, help="Number of CPUs to use for parallel processing")
    parser.add_argument("--max_pending", type=int, default=32, help="Maximum number of pending jobs in the pool")
    
    args = parser.parse_args() 
    if args.max_batches == "None":
        args.max_batches = None
    elif args.max_batches != "None":
        args.max_batches = int(args.max_batches)

    return args

if __name__ == "__main__":
    args = parse_args()

    # # Create converters
    converter_train = CrossdockedNoLinksZarrConverter(output_path=args.train_zarr_output_dir, num_workers=args.n_cpus)
    converter_test = CrossdockedNoLinksZarrConverter(output_path=args.test_zarr_output_dir, num_workers=args.n_cpus)

    
    batches_train= converter_train.get_ligand_receptor_batches_external(
    data=data["train"],
    root_dir=args.root_dir,
    batch_size= args.batch_size, 
    max_num_batches=args.max_batches
    )

    batches_test = converter_test.get_ligand_receptor_batches_external(
    data=data["test"],
    root_dir=args.root_dir,
    batch_size = args.batch_size,
    max_num_batches=args.max_batches
    )

    # # Run the processing serially
    # for batch in batches:
    #     result = converter._process_batch(batch, pocket_cutoff=args.pocket_cutoff, n_cpus=args.n_cpus)
    #     converter._write_system_batch(result)

    converter_train.process_dataset_parallel(
        batches=batches_train,
        pocket_cutoff=args.pocket_cutoff,
        n_cpus=args.n_cpus,           
        max_pending=args.max_pending,
    )

    converter_test.process_dataset_parallel(
        batches=batches_test,
        pocket_cutoff=args.pocket_cutoff,
        n_cpus=args.n_cpus,           
        max_pending=args.max_pending,
    )
    
    #print(converter_train.root.tree())

    #print("Test completed successfully.")
