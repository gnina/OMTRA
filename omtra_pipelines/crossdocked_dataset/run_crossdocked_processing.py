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
import zarr
import threading
import psutil  # For monitoring memory usage
import sys
#logging.basicConfig(level=logging.CRITICAL)
logging.disable(level=logging.CRITICAL)  # Set to CRITICAL to suppress debug output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.vlen_utf8")



# Crossdocked Data Directories
types_file_dir = "/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types"
root_dir = "/net/galaxy/home/koes/paf46_shared/cd2020_v1.3"
zarr_output_dir = "test_output.zarr" #this is not used for the multiple types files
test_types_file = "test.types"

types_files_pairs = [("it2_tt_v1.3_0_train0.types", "it2_tt_v1.3_0_test0.types"),
        ("it2_tt_v1.3_0_train1.types", "it2_tt_v1.3_0_test1.types"),
        ("it2_tt_v1.3_0_train2.types", "it2_tt_v1.3_0_test2.types")]
    

def parse_args():
    parser = argparse.ArgumentParser(description="Test Crossdocked dataset processing")
    parser.add_argument("--cd_directory", type=str, default=types_file_dir, help="Crossdocked types file directory")
    parser.add_argument("--pocket_cutoff", type=float, default=8.0, help="Pocket cutoff distance")
    parser.add_argument("--zarr_output_dir", type=str, default=zarr_output_dir, help="Output Zarr directory for testing")
    parser.add_argument("--root_dir", type=str, default=root_dir, help="Root directory for crossdocked data")
    
    parser.add_argument("--max_batches", type=str, default="None", help="Maximum number of batches to process in each types file")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size of how many receptor-ligand pairs to process in each batch")
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
    
    '''
    ################### ONE TYPES FILE ######################
    types_file_path = "/net/galaxy/home/koes/jmgupta/omtra_2/omtra_pipelines/crossdocked_dataset/test_types_file.types"
    # # Create converter
    output_dir = (args.zarr_output_dir)
    converter = CrossdockedNoLinksZarrConverter(output_path=os.path.join(args.zarr_output_dir, "test.zarr"), num_workers=args.n_cpus)

    batches = converter.get_ligand_receptor_batches_types(
    types_file=types_file_path,
    root_dir=args.root_dir,
    batch_size= args.batch_size, 
    max_num_batches= args.max_batches
    )

    converter.process_dataset_serial(
        batches=batches,
        pocket_cutoff=args.pocket_cutoff,
        n_cpus=1,           
    )
    '''
    cd_directory = Path(args.cd_directory)
    all_types_files = cd_directory.glob("it2_tt_v1.3_0_*.types")

    # Group types files into pairs
    types_files_pairs = []
    for file in all_types_files:
        if "train" in file.name:
            # Grab corresponding test file
            test_file = file.with_name(file.name.replace("train", "test"))
            if test_file.exists():
                types_files_pairs.append((file.name, test_file.name))
    
    # File saving: We create 3 folders: internal_split0, internal_split1, internal_split2
    # Each folder contains train.zarr and val.zarr

    #Types_file pairs is a list of lists, where each inner list contains pairs of train and test types files represented as tuples (train, test)
    output_dir = Path(args.zarr_output_dir)  # Either data/crossdocked or omtra_pipelines/crossdocked_dataset/zarr_storage_test
    i = 0
    for train_file, test_file in types_files_pairs:
        current_folder = f"internal_split{i}"
        #Process training set
        train_types_file_path = os.path.join(args.root_dir, 'types', train_file)
        
        converter_train = CrossdockedNoLinksZarrConverter(
            output_path=str(output_dir / current_folder /f"train.zarr"), 
            num_workers=args.n_cpus
            )

        batches_train = converter_train.get_ligand_receptor_batches_types(
            types_file=train_types_file_path,
            root_dir=args.root_dir,
            batch_size= args.batch_size, 
            max_num_batches= args.max_batches
        )
        
        # converter_train.process_dataset_serial(
        #     batches=batches_train,
        #     pocket_cutoff=args.pocket_cutoff,
        #     n_cpus=1     
        # )
        #### Process in parallel ####
        converter_train.process_dataset_parallel(
        batches=batches_train,
        pocket_cutoff=args.pocket_cutoff,
        n_cpus=args.n_cpus,           
        max_pending=args.max_pending,
        )

        #Process test set
        test_types_file_path = os.path.join(args.root_dir, 'types', test_file)

        converter_test = CrossdockedNoLinksZarrConverter(
            output_path=str(output_dir / current_folder /f"val.zarr"), 
            num_workers=args.n_cpus
        )

        batches_test = converter_test.get_ligand_receptor_batches_types(
            types_file=test_types_file_path,
            root_dir=args.root_dir,
            batch_size= args.batch_size, 
            max_num_batches= args.max_batches
        )

        # converter_test.process_dataset_serial(
        #     batches=batches_test,
        #     pocket_cutoff=args.pocket_cutoff,
        #     n_cpus=1
        # )
        #### Process in parallel ####
        converter_test.process_dataset_parallel(
            batches=batches_test,
            pocket_cutoff=args.pocket_cutoff,
            n_cpus=args.n_cpus,           
            max_pending=args.max_pending,
        )
        #increment i for next pair of types files
        i += 1


    print(converter_train.root.tree())
    print(converter_test.root.tree())

    # Today's zarr files
    # todays_files = [
    #     'test2_output.zarr',
    #     'train2_output.zarr', 
    #     'test1_output.zarr',
    #     'train1_output.zarr',
    #     'test0_output.zarr', 
    #     'train0_output.zarr'
    # ]

    # print("Checking today's zarr files for NPNDE data:")
    # for zarr_file in todays_files:
    #     try:
    #         root = zarr.open_group(zarr_file, mode='r')
    #         npnde_coords_shape = root['npnde']['coords'].shape
    #         npnde_lookup_count = len(root.attrs.get("npnde_lookup", []))
            
    #         print(f"\n{zarr_file}:")
    #         print(f"  NPNDE coords shape: {npnde_coords_shape}")
    #         print(f"  NPNDE lookup entries: {npnde_lookup_count}")
            
    #         if npnde_coords_shape[0] > 0:
    #             print(f"  FOUND NPNDEs!  ")
    #             print(f"  NPNDE coords: {root['npnde']['coords'][:]}")
    #             print(f"  NPNDE atom types: {root['npnde']['atom_types'][:]}")
    #     except Exception as e:
    #         print(f"\n{zarr_file}: Error - {e}")
