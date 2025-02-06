import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import defaultdict
from omtra_pipelines.plinder_dataset.plinder_pipeline import SystemProcessor, StructureData, LigandData

class PlinderZarrConverter:
    def __init__(
        self, 
        output_path: str,
        system_processor: SystemProcessor,
        chunk_size: int = 50
    ):
        self.output_path = Path(output_path)
        self.system_processor = system_processor
        self.chunk_size = chunk_size
        
        self.store = zarr.storage.LocalStore(str(self.output_path))
        self.root = zarr.group(store=self.store)

        self.receptor = self.root.create_group('receptor')
        self.apo = self.root.create_group('apo')
        self.pred = self.root.create_group('pred')
        self.pocket = self.root.create_group('pocket')

        for group in [self.receptor, self.apo, self.pred, self.pocket]:
            group.create_array('coords', shape=(0, 3), chunks=(self.chunk_size, 3), dtype=np.float32)
            group.create_array('atom_names', shape=(0,), chunks=(self.chunk_size,), dtype=str)
            group.create_array('res_ids', shape=(0,), chunks=(self.chunk_size,), dtype=np.int32)
            group.create_array('res_names', shape=(0,), chunks=(self.chunk_size,), dtype=str)
            group.create_array('chain_ids', shape=(0,), chunks=(self.chunk_size,), dtype=str)

            if group != self.pocket:
                group.create_array('cif_paths', shape=(0,), chunks=(self.chunk_size,), dtype=str)
        
        self.ligand = self.root.create_group('ligand')

        self.ligand.create_array('coords', shape=(0, 3), chunks=(self.chunk_size, 3), dtype=np.float32)
        self.ligand.create_array('atom_types', shape=(0,), chunks=(self.chunk_size,), dtype=np.int32)
        self.ligand.create_array('atom_charges', shape=(0,), chunks=(self.chunk_size,), dtype=np.float32)
        self.ligand.create_array('bond_types', shape=(0,), chunks=(self.chunk_size,), dtype=np.int32)
        self.ligand.create_array('bond_indices', shape=(0, 2), chunks=(self.chunk_size, 2), dtype=np.int32)
        self.ligand.create_array('sdf_paths', shape=(0,), chunks=(self.chunk_size,), dtype=str)
        
        
        # Initialize lookup tables
        self.receptor_lookup = []  # [{system_id, idx, start_idx, end_idx}]
        self.apo_lookup = []   # [{system_id, apo_id, holo_idx, apo_idx, start_idx, end_idx}]
        self.pred_lookup = []  # [{system_id, pred_id, holo_idx, pred_idx, start_idx, end_idx}]
        self.ligand_lookup = [] # [{system_id, ligand_id, holo_idx, ligand_idx, ligand_num, atom_start_idx, atom_end_idx, bond_start_idx, bond_end_idx}]
        self.pocket_lookup = [] # [{system_id, pocket_id, holo_idx, pocket_idx, pocket_num, start_idx, end_idx}]

    
    def _append_structure_data(self, group: zarr.Group, data: StructureData) -> tuple[int, int]:
        """
        Append structure data to arrays and return start and end indices.
        
        Args:
            group: zarr group to append to
            data: structure data to append
        
        Returns:
            tuple[int, int]: (start_idx, end_idx) of the appended structure
        """
        current_len = group['coords'].shape[0]
        num_atoms = len(data.coords)
        new_len = current_len + num_atoms
        
        # Resize and append atomic data
        group['coords'].resize((new_len, 3))
        group['coords'][current_len:] = data.coords
        
        group['atom_names'].resize((new_len,))
        group['atom_names'][current_len:] = data.atom_names
        
        group['res_ids'].resize((new_len,))
        group['res_ids'][current_len:] = data.res_ids
        
        group['res_names'].resize((new_len,))
        group['res_names'][current_len:] = data.res_names
        
        group['chain_ids'].resize((new_len,))
        group['chain_ids'][current_len:] = data.chain_ids

        # if data.cif is not None:
           # cif_array = group['cif_paths']
           # new_len = cif_array.shape[0] + 1
           # cif_array.resize((new_len,))
           # cif_array[-1] = data.cif

        return current_len, new_len

    def _append_ligand_data(self, group: zarr.Group, data: LigandData) -> tuple[int, int, int, int]:
        """
        Append ligand data to arrays

        Args:
            group: zarr group to append to
            data: ligand data to append
        
        Returns:
            tuple[int, int, int, int]: (atom_start, atom_end, bond_start, bond_end) of the appended structure
        
        """
        current_len = group['coords'].shape[0]
        num_atoms = len(data.coords)
        new_len = current_len + num_atoms
        
        # Resize and append atomic data
        group['coords'].resize((new_len, 3))
        group['coords'][current_len:] = data.coords
        
        group['atom_types'].resize((new_len,))
        group['atom_types'][current_len:] = data.atom_types
        
        group['atom_charges'].resize((new_len,))
        group['atom_charges'][current_len:] = data.atom_charges
        
        num_bonds = len(data.bond_indices)
        bond_current_len = group['bond_types'].shape[0]
        new_bond_len = bond_current_len + num_bonds
            
        group['bond_types'].resize((new_bond_len,))
        group['bond_types'][-num_bonds:] = data.bond_types
            
        group['bond_indices'].resize((new_bond_len, 2))
        group['bond_indices'][-num_bonds:] = data.bond_indices
        
        # sdf_array = group['sdf_paths']
        # new_sdf_len = sdf_array.shape[0] + 1
        # sdf_array.resize((new_sdf_len,))
        # sdf_array[-1] = data.sdf

        return current_len, new_len, bond_current_len, new_bond_len

    def process_system(self, system_id: str):
        """Process a single system"""
        # Get system data
        system_data = self.system_processor.process_system(system_id)
         
        # Process holo structure
        receptor_idx = len(self.receptor_lookup)
        receptor_start, receptor_end = self._append_structure_data(self.receptor, system_data["receptor"])
        self.receptor_lookup.append({
            'system_id': system_id, 
            'receptor_idx': receptor_idx, 
            'start': receptor_start, 
            'end': receptor_end,
            'ligand_idxs': [],
            'pocket_idxs': [],
            'apo_idxs': None,
            'pred_idxs': None
            })
        
        # Process ligands and their corresponding pockets
        ligand_count = 0
        ligand_idxs, pocket_idxs = [], []
        for ligand_id, ligand_data in system_data["ligands"].items():
            # Process ligand
            ligand_idx = len(self.ligand_lookup)
            ligand_idxs.append(ligand_idx)
            atom_start, atom_end, bond_start, bond_end = self._append_ligand_data(self.ligand, ligand_data)
            self.ligand_lookup.append({
                'system_id': system_id,
                'ligand_id': ligand_id,
                'receptor_idx': receptor_idx,
                'ligand_idx': ligand_idx,
                'ligand_num': ligand_count,
                'atom_start': atom_start,
                'atom_end': atom_end,
                'bond_start': bond_start,
                'bond_end': bond_end
            })
            
            # Process corresponding pocket
            pocket_data = system_data["pockets"][ligand_id]
            pocket_idx = len(self.pocket_lookup)
            pocket_idxs.append(pocket_idx)
            pocket_start, pocket_end = self._append_structure_data(self.pocket, pocket_data)
            self.pocket_lookup.append({
                'system_id': system_id,
                'pocket_id': ligand_id,
                'receptor_idx': receptor_idx,
                'pocket_idx': pocket_idx, # should be 1:1 ligand to pocket correspondance, but just in case
                'pocket_count': ligand_count,
                'start': pocket_start,
                'end': pocket_end
            })
            
            ligand_count += 1

        self.receptor_lookup[-1]['ligand_idxs'] = ligand_idxs
        self.receptor_lookup[-1]['pocket_idxs'] = pocket_idxs   

        # Process apo structures
        if system_data["apo_structures"]:
            apo_idxs = []
            for apo_id, apo_data in system_data["apo_structures"].items():
                apo_idx = len(self.apo_lookup)
                apo_idxs.append(apo_idx)
                apo_start, apo_end = self._append_structure_data(self.apo, apo_data)
                self.apo_lookup.append({
                    'system_id': system_id,
                    'apo_id': apo_id,
                    'receptor_idx': receptor_idx,
                    'apo_idx': apo_idx,
                    'start': apo_start,
                    'end': apo_end
                })
            self.receptor_lookup[-1]['apo_idxs'] = apo_idxs
        
        # Process pred structures
        if system_data["pred_structures"]:
            pred_idxs = []
            for pred_id, pred_data in system_data["pred_structures"].items():
                pred_idx = len(self.pred_lookup)
                pred_idxs.append(pred_idx)
                pred_start, pred_end = self._append_structure_data(self.pred, pred_data)
                self.pred_lookup.append({
                    'system_id': system_id,
                    'pred_id': pred_id,
                    'receptor_idx': receptor_idx,
                    'pred_idx': pred_idx,
                    'start': pred_start,
                    'end': pred_end
                })
            self.receptor_lookup[-1]['pred_idxs'] = pred_idxs

    def process_dataset(self, system_ids: List[str]):
        """Process list of systems"""
        for system_id in tqdm(system_ids, desc="Processing systems"):
            try:
                self.process_system(system_id)
            except Exception as e:
                print(f"Error processing system {system_id}: {e}")
                continue
        
        # Store lookup tables as attributes
        self.root.attrs['receptor_lookup'] = self.receptor_lookup
        self.root.attrs['apo_lookup'] = self.apo_lookup
        self.root.attrs['pred_lookup'] = self.pred_lookup
        self.root.attrs['ligand_lookup'] = self.ligand_lookup
        self.root.attrs['pocket_lookup'] = self.pocket_lookup

def load_lookups(zarr_path: str = None, root: zarr.Group = None) -> Dict[str, pd.DataFrame]:
    """Helper function to load lookup tables from a zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)
    
    lookups = {}
    for key in ['receptor_lookup', 'apo_lookup', 'pred_lookup', 'ligand_lookup', 'pocket_lookup']:
        if key in root.attrs:
            data_type = key[:-7]
            lookups[data_type] = pd.DataFrame(root.attrs[key])
    
    return lookups

def get_receptor(idx: int, zarr_path: str = None, root: zarr.Group = None) -> StructureData:
    """Helper function to load receptor from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    receptor_df = pd.DataFrame(root.attrs['receptor_lookup'])
    receptor_info = receptor_df[receptor_df['receptor_idx'] == idx].iloc[0]

    start, end = receptor_info['start'], receptor_info['end']

    receptor = StructureData(
        coords = root['receptor']['coords'][start:end],
        atom_names = root['receptor']['atom_names'][start:end],
        res_ids = root['receptor']['res_ids'][start:end],
        res_names = root['receptor']['res_names'][start:end],
        chain_ids = root['receptor']['chain_ids'][start:end],
        # cif = root['receptor']['cif_paths'][idx]
    )
    return receptor 

def get_ligand(lig_idx: int, zarr_path: str = None, root: zarr.Group = None) -> LigandData:
    """Helper function to load ligand from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    ligand_df = pd.DataFrame(root.attrs['ligand_lookup'])
    ligand_info = ligand_df[ligand_df['ligand_idx'] == lig_idx].iloc[0]

    atom_start, atom_end = ligand_info['atom_start'], ligand_info['atom_end']
    bond_start, bond_end = ligand_info['bond_start'], ligand_info['bond_end']

    ligand = LigandData(
        # sdf = root['ligand']['sdf_paths'][lig_idx],
        sdf = "",
        coords = root['ligand']['coords'][atom_start:atom_end],
        atom_types = root['ligand']['atom_types'][atom_start:atom_end] ,
        atom_charges = root['ligand']['atom_charges'][atom_start:atom_end],
        bond_types = root['ligand']['bond_types'][bond_start:bond_end],
        bond_indices = root['ligand']['bond_indices'][bond_start:bond_end]
    )

    return ligand

def get_pocket(pocket_idx: int, zarr_path: str = None, root: zarr.Group = None) -> StructureData:
    """Helper function to load pocket from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    pocket_df = pd.DataFrame(root.attrs['pocket_lookup'])
    pocket_info = pocket_df[pocket_df['pocket_idx'] == pocket_idx].iloc[0]

    start, end = pocket_info['start'], pocket_info['end']

    pocket = StructureData(
        coords = root['pocket']['coords'][start:end],
        atom_names = root['pocket']['atom_names'][start:end],
        res_ids = root['pocket']['res_ids'][start:end],
        res_names = root['pocket']['res_names'][start:end],
        chain_ids = root['pocket']['chain_ids'][start:end]
    )

    return pocket

def get_apo(apo_idx: int, zarr_path: str = None, root: zarr.Group = None) -> StructureData:
    """Helper function to load apo structure from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    apo_df = pd.DataFrame(root.attrs['apo_lookup'])
    apo_info = apo_df[apo_df['apo_idx'] == apo_idx].iloc[0]

    start, end = apo_info['start'], apo_info['end']

    apo = StructureData(
        coords = root['apo']['coords'][start:end],
        atom_names = root['apo']['atom_names'][start:end],
        res_ids = root['apo']['res_ids'][start:end],
        res_names = root['apo']['res_names'][start:end],
        chain_ids = root['apo']['chain_ids'][start:end],
        # cif = root['apo']['cif_paths'][apo_idx]
    )

    return apo 

def get_pred(pred_idx: int, zarr_path: str = None, root: zarr.Group = None) -> StructureData:
    """Helper function to load predicted structure from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    pred_df = pd.DataFrame(root.attrs['pred_lookup'])
    pred_info = pred_df[pred_df['pred_idx'] == pred_idx].iloc[0]

    start, end = pred_info['start'], pred_info['end']

    pred = StructureData(
        coords = root['pred']['coords'][start:end],
        atom_names = root['pred']['atom_names'][start:end],
        res_ids = root['pred']['res_ids'][start:end],
        res_names = root['pred']['res_names'][start:end],
        chain_ids = root['pred']['chain_ids'][start:end],
        # cif = root['pred']['cif_paths'][pred_idx]
    )
    
    return pred

def get_system(zarr_path: str, receptor_idx: int) -> Dict:
    """Helper function to load system from zarr store"""
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.group(store=store)


    lookups = load_lookups(root=root)

    receptor_df = lookups['receptor']
    ligand_df = lookups['ligand']
    apo_df = lookups['apo']
    pred_df = lookups['pred']
    pocket_df = lookups['pocket']

    receptor_info = receptor_df[receptor_df['receptor_idx'] == receptor_idx].iloc[0]

    receptor = get_receptor(receptor_idx, root=root)
    ligands = {}
    for lig_idx in receptor_info['ligand_idxs']:
        ligand_info = ligand_df[ligand_df['ligand_idx'] == lig_idx].iloc[0]
        ligands[ligand_info['ligand_id']] = get_ligand(lig_idx, root=root)
    
    pockets = {}
    for pocket_idx in receptor_info['pocket_idxs']:
        pocket_info = pocket_df[pocket_df['pocket_idx'] == pocket_idx].iloc[0]
        pockets[pocket_info['pocket_id']] = get_pocket(pocket_idx, root=root)
    
    apos = None
    if receptor_info['apo_idxs'] is not None:
        apos = {}
        for apo_idx in receptor_info['apo_idxs']:
            apo_info = apo_df[apo_df['apo_idx'] == apo_idx].iloc[0]
            apos[apo_info['apo_id']] = get_apo(apo_idx, root=root)

    preds = None
    if receptor_info['pred_idxs'] is not None:
        preds = {}
        for pred_idx in receptor_info['pred_idxs']:
            pred_info = pred_df[pred_df['pred_idx'] == pred_idx].iloc[0]
            preds[pred_info['pred_id']] = get_pred(pred_idx, root=root)
    
    return {
        'receptor': receptor,
        'ligands': ligands,
        'pockets': pockets,
        'apo_structures': apos,
        'pred_structures': preds
    }