import os
from pathlib import Path
from omtra.dataset.crossdocked import CrossdockedDataset
from rdkit import Chem
import logging
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time
import traceback
import sys
import omtra.load.quick as quick_load
from omtra.utils import omtra_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/crossdocked/internal_split0", help="Directory containing crossdocked data not including omtra root")
    parser.add_argument("--ckpt", type=str, default="/net/galaxy/home/koes/jmgupta/omtra_2/outputs/2025-08-25/crossdocked_split0_lig_extrafeats_pos_enc_2025-08-25_22-06-597783/checkpoints/last.ckpt", help="Checkpoint of model")
    parser.add_argument("--split", type=str, default="val", help="Split to load: train, val")
    '''
    parser.add_argument("--root_dir", type=str, default="/net/galaxy/home/koes/jmgupta/OMTRA/omtra_pipelines/crossdocked_dataset/zarr_data_with_npnde", help="Root directory for crossdocked data")
    #parser.add_argument("--link_version", type=str, default="", help="Optional link version (e.g., 'apo', 'pred')")
    #parser.add_argument("--split", type=str, default="internal_split0/train", help="Split name and zarr file name")
    #parser.add_argument("--n_samples", type=int, default=5, help="number of systems to load")
    '''
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    '''
    # set zarr path
    #zarr_path = f"{args.root_dir}/{args.split}" #split should be for example train_zarr/train0_output.zarr (folder name and file name)
    # Initialize the CrossdockedDataset
    dataset = CrossdockedDataset(
        split="test", # for testing, use args.split otherwise
        processed_data_dir=args.root_dir,
    )
    '''
    ckpt = Path(args.ckpt)

    dataset = CrossdockedDataset(
    split = args.split, 
    processed_data_dir = "/net/galaxy/home/koes/jmgupta/omtra_2/data/crossdocked/internal_split0"
    )

    cfg_file = ckpt.parent.parent / '.hydra/config.yaml'
    cfg = quick_load.load_trained_model_cfg(cfg_file)
    crossdocked_path = Path(omtra_root()) / args.data_dir
    cfg.crossdocked_path = str(crossdocked_path)

    model = quick_load.omtra_from_checkpoint(str(ckpt)).eval()
    dm = quick_load.datamodule_from_config(cfg)
    multiset = dm.load_dataset(args.split)
    dataset = multiset.datasets['crossdocked']

    print(f"Loading ligand extra features ")
    print(dataset.system_lookup.columns)
    print(dataset.system_lookup.head())
    #get item method
    graph = dataset[("rigid_docking_condensed", 2567)]
    #get extra features
    # system_info = dataset.system_lookup.iloc[2567]
    # lig_atom_start, lig_atom_end = int(system_info["lig_atom_start"]), int(system_info["lig_atom_end"])

    # ligand_atom_types = dataset.slice_array("ligand/atom_types", lig_atom_start, lig_atom_end)
    # ligand_atom_charges = dataset.slice_array("ligand/atom_charges", lig_atom_start, lig_atom_end)
    # lig_extra_feats = dataset.slice_array('ligand/extra_feats', lig_atom_start, lig_atom_end)
    # lig_extra_feats = lig_extra_feats[:, :-1] #everything but fragments


    
    



    '''
    graph = dataset[("protein_ligand_denovo", 0)]
    print("Node types:", graph.ntypes)
    print("Edge types:", graph.etypes)

    # Search for a sample with npnde_to_npnde edges
    # Search specifically for systems with npndes
    print("Searching for systems with npndes...")

    systems_with_npndes = dataset.system_lookup[
        dataset.system_lookup['npnde_idxs'].notna() & 
        (dataset.system_lookup['npnde_idxs'] != '[]')
    ]['system_idx'].tolist()

    print(f"Found {len(systems_with_npndes)} systems with npndes")
    print(f"First few system indices with npndes: {systems_with_npndes[:10]}")

    # Test the first system with npndes
    if systems_with_npndes:
        test_idx = systems_with_npndes[0]
        print(f"\nTesting system {test_idx} (known to have npndes)")
        
        graph = dataset[("protein_ligand_denovo", test_idx)]
        npnde_edges = graph.edges['npnde_to_npnde']
        
        print(f"npnde_to_npnde edges: {npnde_edges.data['e_1_true'].shape[0]}")
        print("Edge data keys:", list(npnde_edges.data.keys()))
        
        for key in npnde_edges.data.keys():
            print(f"  {key}: shape {npnde_edges.data[key].shape}")
    '''