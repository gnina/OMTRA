# Import libraries
import re, subprocess, os, gzip, json, glob, multiprocessing
import logging
import os
import io
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import time
import gc

import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite.structure.io.pdb import PDBFile
import biotite.structure.io as strucio
from biotite.structure.io import save_structure
from biotite.interface.rdkit import to_mol

from biotite.structure import molecule_iter, filter_amino_acids, stack
from omtra.data.extra_ligand_features import *

from omtra.data.plinder import (
    LigandData,
    PharmacophoreData,
    StructureData,
    SystemData,
    BackboneData,
)
from omtra.data.pharmacophores import get_pharmacophores
from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.misc import bad_mol_reporter
from omtra.constants import lig_atom_type_map, npnde_atom_type_map, aa_substitutions, residue_to_single
from omtra_pipelines.plinder_dataset.utils import _DEFAULT_DISTANCE_RANGE, setup_logger
from omtra_pipelines.plinder_dataset.filter import filter
from plinder.core import PlinderSystem

from rdkit import Chem
from rdkit.Chem import SanitizeFlags

import torch
import tempfile

#from line_profiler import profile
'''
ESM Stuff
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
    )
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

#Sets up using the ESM3 model to return embedding vectors from protein sequences
EMBEDDING_CONFIG = LogitsConfig(
    sequence=False, return_embeddings=True, return_hidden_states=False
)
'''
# Enables logging for tracking messages during script execution
logger = setup_logger(
    __name__,
)
logging.getLogger().setLevel(logging.CRITICAL)
# Class for writing molecular structures to a pdb file
# !!: Might require modification; see below
class PDBWriter:
    def __init__(self, chain_mapping: Optional[Dict[str, str]] = None):
        self.chain_mapping = chain_mapping
    
    #Read in structural data, constructs representation of the molecule using biotite.structure
    # !!: Might need to modify the datatype, StructureData is coming from omtra.data.plinder
    # !!: source code for data type at OMTRA/omtra/data/plinder/_init_.py
    def write(self, struct_data: StructureData, output_path: str):
        logger.info("Writing structure to %s", output_path)
        struct = struc.AtomArray(len(struct_data.coords))

        struct.coord = struct_data.coords
        struct.atom_name = struct_data.atom_names
        struct.res_id = struct_data.res_ids
        struct.res_name = struct_data.res_names
        struct.hetero = np.full(len(struct_data.coords), False)

        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(struct)
        pdb_file.write(output_path)

# System is a ligand/receptor system
class SystemProcessor:
    #!!: defining parameters
    # modifications are adding receptor and ligand file path, removing link type
    def __init__(
        self,
        receptor_path: str, #.gninatypes
        ligand_path: str, #.gninatypes
        ligand_atom_map: List[str] = lig_atom_type_map,
        npnde_atom_map: List[str] = npnde_atom_type_map,
        pocket_cutoff: float = 8.0, # 8.0 is default, but can be set to any value
        n_cpus: int = 1, # 1 is default, but can be set to number of available CPUs
        raw_data: str = "/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types"  #!!: double check we want to route to types
    ):
        logger.debug("Initializing StructureProcessor with cutoff=%f", pocket_cutoff)
        self.ligand_atom_map = ligand_atom_map
        self.npnde_atom_map = npnde_atom_map
        self.pocket_cutoff = pocket_cutoff
        
        self.ligand_tensorizer = MoleculeTensorizer(
            atom_map=ligand_atom_map, n_cpus=1 #!!: set to 1 for now, can be changed later
        )
        
        self.npnde_tensorizer = MoleculeTensorizer(
            atom_map=npnde_atom_map, n_cpus=1
        )

        self.raw_data = Path(raw_data) #!!: this is now going to be types files
        
        #Adjust receptor/ligand file names
        receptor_path = Path(receptor_path)
        ligand_path = Path(ligand_path)

        # if receptor path is .gninatypes, convert to pdb
        # if ligand path is .gninatypes, convert to sdf.gz
        if receptor_path.suffix == ".gninatypes" and ligand_path.suffix == ".gninatypes":
            receptor_path, ligand_path = self.adjust_gninatypes_filepaths(receptor_path, ligand_path)
        elif receptor_path.suffix == ".pdb" and ligand_path.suffix == ".sdf":
            receptor_path, ligand_path = self.adjust_pdb_sdf_filepaths(receptor_path, ligand_path)

        self.receptor_path = Path(receptor_path)
        self.ligand_path = Path(ligand_path)
        logger.info(f"Receptor/Ligand file paths remapped")

        
        #self.link_type = link_type
        self.pdb_writer = None

    #extract_backbone converts a protein's backbone atoms from an AtomArray to the BackboneData format
    # !!: are we still using the BackboneData format? coming from omtra.data.plinder
    #@profile
    # def extract_backbone_new(self, backbone: struc.AtomArray) -> BackboneData:
    #     # Map: (chain_id, res_id) -> [N, CA, C] coords
    #     coords_dict = {}  # key: (chain_id, res_id), value: np.array of shape (3, 3)
    #     res_meta = {}     # key: (chain_id, res_id), value: {'res_name', 'res_id', 'chain_id'}

    #     atom_index_map = {"N": 0, "CA": 1, "C": 2}

    #     for i in range(len(backbone)):
    #         chain_id = backbone.chain_id[i]
    #         res_id = backbone.res_id[i]
    #         atom_name = backbone.atom_name[i]

    #         if atom_name not in atom_index_map:
    #             continue

    #         key = (chain_id, res_id)
    #         atom_idx = atom_index_map[atom_name]

    #         if key not in coords_dict:
    #             coords_dict[key] = np.full((3, 3), np.nan)

    #         # Only set coord if not already set (mimics res_atoms.coord[mask][0])
    #         if np.isnan(coords_dict[key][atom_idx]).all(): # checks if all coordinates are NaN
    #             coords_dict[key][atom_idx] = backbone.coord[i]

    #         # Store metadata from first occurrence only
    #         if key not in res_meta:
    #             res_meta[key] = {
    #                 "res_name": backbone.res_name[i],
    #                 "res_id": res_id,
    #                 "chain_id": chain_id,
    #             }

    #     # Sort residue keys lexicographically to match np.unique(compound_keys)
    #     sorted_keys = sorted(coords_dict.keys(), key=lambda x: f"{x[0]}_{x[1]}")

    #     coords_list = []
    #     res_ids = []
    #     res_names = []
    #     chain_ids = []

    #     for key in sorted_keys:
    #         coord = coords_dict[key]
    #         if np.isnan(coord).any():
    #             logger.warning(f"Missing backbone atom in residue {key}")
    #             return None  # Abort like original function

    #         coords_list.append(coord)
    #         res_ids.append(res_meta[key]["res_id"])
    #         res_names.append(res_meta[key]["res_name"])
    #         chain_ids.append(res_meta[key]["chain_id"])

    #     # Convert to arrays
    #     coords = np.array(coords_list)           # (num_residues, 3, 3)
    #     res_ids = np.array(res_ids, dtype=int)
    #     res_names = np.array(res_names)
    #     chain_ids = np.array(chain_ids)

    #     return BackboneData(
    #         coords=coords,
    #         res_ids=res_ids,
    #         res_names=res_names,
    #         chain_ids=chain_ids,
    #     )

    #########Original below##########
    def extract_backbone(
        self,
        backbone: struc.AtomArray,
    ) -> BackboneData:
    # creating compound keys of chain and residue ID
        compound_keys = np.array(
            [f"{chain}_{res}" for chain, res in zip(backbone.chain_id, backbone.res_id)]
        )
    #get unique residues
        unique_compound_keys = np.unique(compound_keys)
        num_residues = len(unique_compound_keys)

        coords = np.zeros((num_residues, 3, 3)) #for each residue, we get atom type (from 3 types) & xyz coordinates for each atom type
        res_ids = np.zeros(num_residues, dtype=int)
        res_names_list = []
        chain_ids_list = []

        for i, compound_key in enumerate(unique_compound_keys):
            chain_id, res_id = compound_key.split("_")
            res_id = int(res_id)

            res_mask = (backbone.chain_id == chain_id) & (backbone.res_id == res_id)
            res_atoms = backbone[res_mask]

            res_ids[i] = res_id
            res_names_list.append(res_atoms.res_name[0])
            chain_ids_list.append(chain_id)

            # for the 3 backbone atoms, geet the coordinates
            for j, atom_name in enumerate(["N", "CA", "C"]):
                atom_mask = res_atoms.atom_name == atom_name
                if np.any(atom_mask):
                    coords[i, j] = res_atoms.coord[atom_mask][0]
                else:
                    logger.warning(f"Error with backbone extraction")
                    return None

        res_names = np.array(res_names_list)
        chain_ids = np.array(chain_ids_list)

        backbone_data = BackboneData(
            coords=coords,
            res_ids=res_ids,
            res_names=res_names,
            chain_ids=chain_ids,
        )
        return backbone_data 

    #@profile
    # process_receptor takes raw receptor file (AtomArray) and processes it into a StructureData object
    def process_receptor(
        self,
        receptor: struc.AtomArray,
        receptor_path: str, #changed to receptor path
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> StructureData:
        #removing waters and hydrogens
        receptor = receptor[receptor.res_name != "HOH"]
        receptor = receptor[receptor.element != "H"]
        receptor = receptor[receptor.element != "D"]
        
        try:
            raw_structure = Path(receptor_path).relative_to(self.raw_data)
        except ValueError:
            raw_structure = Path(receptor_path).name

        #if chain mapping is provided, map chain IDs to values as defined
        if chain_mapping is not None:
            chain_ids = [chain_mapping.get(chain, chain) for chain in receptor.chain_id]
            receptor.chain_id = chain_ids
        logger.info(f"Processing receptor from: {receptor_path}")
        #extract backbone
        backbone = receptor[struc.filter_peptide_backbone(receptor)]
        backbone_data = self.extract_backbone(backbone)

        if backbone_data is None:
            return None
        #check backbone atom ordering
        receptor = self.check_backbone_order(receptor)
        if receptor is None:
            return None
        #create boolean mask thats true to backbone atoms, false otherwise
        bb_mask = struc.filter_peptide_backbone(receptor)
        return StructureData(
            cif=str(raw_structure), #note: raw_structure here is a pdb NOT cif
            coords=receptor.coord,
            atom_names=receptor.atom_name,
            elements=receptor.element,
            res_ids=receptor.res_id,
            res_names=receptor.res_name,
            chain_ids=receptor.chain_id,
            backbone_mask=bb_mask,
            backbone=backbone_data,
        )
    # check_backbone_order ensures that for each residue, the three backbone atoms are in the expected order
    # in the AtomArray
    #@profile
    def check_backbone_order(self, receptor: struc.AtomArray) -> struc.AtomArray:
        unique_residues = list(set(zip(receptor.chain_id, receptor.res_id)))
        reordering_needed = False

        #for each unique residue
        for chain_id, res_id in unique_residues:
            #create a mask for all atoms in current residue
            residue_mask = (receptor.chain_id == chain_id) & (receptor.res_id == res_id)
            residue_atoms = receptor[residue_mask]
            #get positions of N, CA, C backbone atoms within the residue
            n_indices = np.where(residue_atoms.atom_name == "N")[0]
            ca_indices = np.where(residue_atoms.atom_name == "CA")[0]
            c_indices = np.where(residue_atoms.atom_name == "C")[0]
            #If missking backbone atoms, skip residue
            if len(n_indices) == 0 or len(ca_indices) == 0 or len(c_indices) == 0:
                continue
            #Get the global indices in the AtomArray
            full_indices = np.where(residue_mask)[0]
            n_idx = full_indices[n_indices[0]]
            ca_idx = full_indices[ca_indices[0]]
            c_idx = full_indices[c_indices[0]]
            #check ordering of atoms, if needed reorder
            if not (n_idx < ca_idx < c_idx):
                reordering_needed = True
                break

        if reordering_needed:
            logger.warning(f"Backbone atom reordering required")
            return self.reorder_backbone_atoms(receptor, unique_residues)
        else:
            return receptor

    # reorder_backbone_atoms fixes residue-level atom order
    #@profile
    def reorder_backbone_atoms(
        self, receptor: struc.AtomArray, unique_residues
    ) -> struc.AtomArray:
        reordered_atoms = []

        for chain_id, res_id in unique_residues:
            residue_mask = (receptor.chain_id == chain_id) & (receptor.res_id == res_id)
            residue_atoms = receptor[residue_mask]
            #get backbone atoms for each residue
            n_mask = residue_atoms.atom_name == "N"
            ca_mask = residue_atoms.atom_name == "CA"
            c_mask = residue_atoms.atom_name == "C"
            backbone_mask = n_mask | ca_mask | c_mask
            #identify index of backbone atoms in residue
            n_idx = np.where(n_mask)[0][0] if np.any(n_mask) else -1
            ca_idx = np.where(ca_mask)[0][0] if np.any(ca_mask) else -1
            c_idx = np.where(c_mask)[0][0] if np.any(c_mask) else -1

            new_order = []
            #atoms before N
            if n_idx > 0:
                new_order.extend(list(range(n_idx)))
            # backbone atom N
            if n_idx != -1:
                new_order.append(n_idx)
            # atoms between N and CA
            if n_idx != -1 and ca_idx != -1:
                for i in range(n_idx + 1, ca_idx):
                    if not backbone_mask[i]:
                        new_order.append(i)
            # backbone atom CA
            if ca_idx != -1:
                new_order.append(ca_idx)
            # backbone atoms between CA and C
            if ca_idx != -1 and c_idx != -1:
                for i in range(ca_idx + 1, c_idx):
                    if not backbone_mask[i]:
                        new_order.append(i)
            # backbone atom C
            if c_idx != -1:
                new_order.append(c_idx)
            # backbone atoms after C
            if c_idx != -1 and c_idx < len(residue_atoms) - 1:
                new_order.extend(range(c_idx + 1, len(residue_atoms)))

            for idx in new_order:
                reordered_atoms.append(residue_atoms[idx])

        if reordered_atoms:
            return struc.stack(reordered_atoms)
        else:
            logger.warning(f"Failed to reorder backbone")
            return None
    
    #process ligands from RDKit to tensors
    #!!: modified this to take ligand path below
    #@profile
    def process_ligand(
        self, ligand_mol: Chem.rdchem.Mol, receptor_array: Chem.rdchem.Mol, ligand_path: str # ligand_mol is an rdkit mol
    ) -> Tuple[LigandData, PharmacophoreData]: #output is a LigandData object
        logger.info("Processing ligand to tensor format")
        
        (xace_mols, failed_idxs, _, _) = ( #got rid of failure counts ad tcv_counts as we have 1 ligand
            self.ligand_tensorizer.featurize_molecules([ligand_mol]) #wrap ligand_mol in list for this function
        )

        #pharmacophore data
        #ligand only pharmacophore extraction
        #convert receptor array to rdkit mol for getting pharmacophores using biotites to mol
        receptor_mol = Chem.MolFromPDBFile(self.receptor_path)
        if not receptor_mol:
            receptor_mol = Chem.MolFromPDBFile(self.receptor_path, sanitize=False)

        P, X, V, I = get_pharmacophores(mol=ligand_mol, rec=receptor_mol)
        if not np.isfinite(V).all():
            logger.warning(
                f"Non-finite pharmacophore vectors found in ligand"
            )
            bad_mol_reporter(
                ligand_mol, 
                note="Pharmacophore vectors contain non-finite values"
            )
        if len(I) != len(P):
            logger.warning(
                f"Length mismatch with interactions {len(I)} and pharm centers {len(P)} in ligand"
            )
            bad_mol_reporter(
                ligand_mol, 
                note="Length mismatch interactions/pharm center"
            )
            failed_mol = ligand_mol
        
        pharmacophores_data = PharmacophoreData(
            coords=P, types=X, vectors=V, interactions=I
        )
        logger.info("Extracted pharmacophore data")
        #Calculate extra ligand features
        extra_lig_features = ligand_properties(ligand_mol)

        lig_fragments = fragment_molecule(ligand_mol)

        logger.info("Extracted extra ligand features")

        ligand_data = LigandData(
                sdf=ligand_path,
                ccd=None, #not defining this
                coords=np.array(xace_mols[0].x, dtype=np.float32), #1 ligand so indices are 0 #positions
                atom_types=xace_mols[0].a, #atom_types
                atom_charges=xace_mols[0].c, #atom_charges
                bond_types=xace_mols[0].e, #bond_types
                bond_indices=xace_mols[0].edge_idxs, #
                is_covalent=False,
                linkages=None,
                #extra feats
                atom_impl_H=extra_lig_features[:,0],
                atom_aro=extra_lig_features[:,1],
                atom_hyb=extra_lig_features[:,2],
                atom_ring=extra_lig_features[:,3],
                atom_chiral=extra_lig_features[:,4],
                fragments=lig_fragments
            )
        logger.info("LigandData Object created.")
        return (ligand_data, pharmacophores_data)
    
    #process non-protein molecules and creates ligand data entries for each one
    #!!: uncertain about how npndes can be saved? like an sdf file also?
    #@profile
    def process_npndes(
        self, npnde_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Dict[str, LigandData]:
        #unpack dictionary of these molecules
        keys = list(npnde_mols.keys())
        mols = list(npnde_mols.values())

        (xace_mols, failed_idxs, failure_counts, tcv_counts) = (
            self.npnde_tensorizer.featurize_molecules(mols)
        )
        
        for i in failed_idxs:
            logger.warning("Failed to tensorize npnde %s", keys[i])
        #get successfully tensorized molecules
        npnde_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        npnde_data = {}
        for i, key in enumerate(npnde_keys):
            sdf = None
            ccd = None
            is_covalent = False
            linkages = None

            npnde_data[key] = LigandData(
                sdf=sdf,
                ccd=ccd,
                coords=np.array(xace_mols[i].x, dtype=np.float32),
                atom_types=xace_mols[i].a,                              
                atom_charges=xace_mols[i].c,                            
                bond_types=xace_mols[i].e,                              
                bond_indices=xace_mols[i].edge_idxs,                    
                is_covalent=is_covalent,
                linkages=linkages,
            )

        return npnde_data

    # remaps atom types in LigandData from ligand atom vocab to npnde atom voca
    #@profile
    def convert_npnde_map(self, ligand: LigandData) -> LigandData:
        atom_types = [self.ligand_atom_map[i] for i in ligand.atom_types]
        new_atom_types = [self.npnde_atom_map.index(atom) for atom in atom_types]
        npnde = LigandData(
            sdf=ligand.sdf,
            ccd=ligand.ccd,
            coords=ligand.coords,
            atom_types=np.array(new_atom_types, dtype=np.int32),
            atom_charges=ligand.atom_charges,
            bond_types=ligand.bond_types,
            bond_indices=ligand.bond_indices,
            is_covalent=ligand.is_covalent,
            linkages=ligand.linkages,
        )
        return npnde
  
    #extract_pocket takes a receptor structure (biotite AtomArray), 3D coordinates of ligand atoms, and optional dict
    #mapping chain IDs, and outputs the part of the receptor that lies within a cutoff distance from the ligand
    #@profile
    def extract_pocket(
        self,
        receptor: struc.AtomArray,
        ligand_coords: np.ndarray,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> StructureData:
        logger.debug("Extracting pocket")

        #remove water and hydrogen atoms
        receptor = receptor[receptor.res_name != "HOH"]
        receptor = receptor[receptor.element != "H"]
        receptor_cell_list = struc.CellList(receptor, cell_size=self.pocket_cutoff)
        #get close atoms
        close_atom_indices = []
        for lig_coord in ligand_coords:
            indices = receptor_cell_list.get_atoms(lig_coord, radius=self.pocket_cutoff)
            close_atom_indices.extend(indices)
        #find unique residue-chain pairs for every atom close to the ligand, grab residue and chain ids)
        close_res_ids = receptor.res_id[close_atom_indices]
        close_chain_ids = receptor.chain_id[close_atom_indices]
        unique_res_pairs = set(zip(close_res_ids, close_chain_ids))
        #get all atoms from close residues
        pocket_indices = []
        for res_id, chain_id in unique_res_pairs:
            res_mask = (receptor.res_id == res_id) & (receptor.chain_id == chain_id)
            res_indices = np.where(res_mask)[0]
            pocket_indices.extend(res_indices)

        if len(pocket_indices) == 0:
            return None
        #if defined, update chain IDs to match a unified naming scheme
        if chain_mapping is not None:
            chain_ids = [chain_mapping.get(chain, chain) for chain in receptor.chain_id]
            receptor.chain_id = chain_ids
        #build AtomArray pocket and extract the backbone
        logger.info(f"Found {len(pocket_indices)} atoms in pocket region")
        pocket = receptor[pocket_indices]
        backbone = pocket[struc.filter_peptide_backbone(pocket)]
        backbone_data = self.extract_backbone(backbone)

        pocket = self.check_backbone_order(pocket)
        if pocket is None:
            return None
        #get backbone mask
        bb_mask = struc.filter_peptide_backbone(pocket)

        #embedding = self.ESM3_embed(pocket.res_name, pocket.chain_id, backbone_data.coords, bb_mask)

        return StructureData(
            coords=pocket.coord,
            atom_names=pocket.atom_name,
            elements=pocket.element,
            res_ids=pocket.res_id,  # original residue ids
            res_names=pocket.res_name,
            chain_ids=pocket.chain_id,
            backbone_mask=bb_mask,
            backbone=backbone_data,
            #pocket_embedding=embedding,
            pocket_embedding= None,
        )
    
    #Returns true if all elements in molecule are protein, false if otherwise
    #@profile
    def is_protein(self, mol_array) -> bool:
        mask = struc.filter_amino_acids(mol_array)
        
        #count number of amino acid atoms
        num_aa_atoms = sum(mask)

        #check if all the atoms are amino acids
        flag = False
        if num_aa_atoms == len(mol_array):
            flag = True
        return flag

    # Looks and filters npndes from a receptor_array, returns cleaned receptor array and list of npndes file paths
    #@profile
    def filter_npndes(self, receptor_array):
        npnde_sdf_strings = []

        protein_mask = np.zeros(len(receptor_array), dtype=bool)
        current_idx = 0
        npnde_count = 0
        npnde_mols = {}
        
        # Collect all protein molecules first
        for molecule in molecule_iter(receptor_array):
            molecule_len = len(molecule) #how many atoms in the molecule

            if self.is_protein(molecule):

                # Mark this molecule's atoms as "keep" in the mask
                protein_mask[current_idx:current_idx + molecule_len] = True
            
            else:
                # #Process NPNDE
                ##### This is faster? #####
                npnde_count += 1
                npnde_mol = to_mol(molecule)
                if npnde_mol is not None:
                    npnde_mols[f"npnde_{npnde_count}"] = npnde_mol
                del molecule
            
            current_idx += molecule_len
        
        #Apply mask to keep only protein atoms
        cleaned_receptor = receptor_array[protein_mask]
        del protein_mask
        del receptor_array

        return (cleaned_receptor, npnde_mols)

    #process_system loads ligands and npndes
    #@profile
    def process_system(self, save_pockets: bool = False) -> Dict[str, Any]:
        
        logger.info("Ligand and Receptor paths: %s, %s", self.ligand_path, self.receptor_path)
        logger.info("Processing system")
        
        ######## Load ligand molecule from .sdf.gz
        try:
            with gzip.open(self.ligand_path, 'rb') as f:
                suppl = Chem.ForwardSDMolSupplier(f, sanitize=False)
                ligand_mol = next(suppl)
        except Exception as e:
            logger.error(f"Failed to load ligand from: {self.ligand_path} — {e}")
            return None

        if not ligand_mol:
            logger.error(f"Ligand molecule is None: {self.ligand_path}")
            return None

        logger.info("Successfully loaded ligand molecule")
        
        ###### Load the receptor molecule from pdb into biotite array
        pdb_file = PDBFile.read(self.receptor_path)
        receptor_array = pdb_file.get_structure(model=1)
        receptor_array.bonds = struc.connect_via_residue_names(receptor_array)
        
        ######  Find & remove npndes
        cleaned_receptor_array, npnde_mols = self.filter_npndes(receptor_array)
        
        result = self.process_structures(
            ligand_mol=ligand_mol, #no longer passing a dictionary
            receptor_array=cleaned_receptor_array,
            npnde_mols=npnde_mols,
            chain_mapping=None,
            save_pockets=save_pockets,
        )

        if not result:
            logger.warning(
                "Skipping due to no ligands remaining"
            )
            return None

        #check if ligand_mol has more than 6 atoms
        if ligand_mol.GetNumAtoms() < 6:
            logger.warning(
                f"Skipping due to small ligand size: {ligand_mol.GetNumAtoms()} atoms"
            )
            return None

        return result
    #process_structure_no_links processes receptor, extracts pockets, saves pockets as PDBs, and creates a list of 
    #systemdata objects to pass to model training
    #@profile
    def process_structures_no_links(
        self,
        ligand_data: LigandData,
        receptor_array: struc.AtomArray,
        pharmacophore_data: PharmacophoreData,
        npnde_data: Optional[Dict[str, LigandData]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> SystemData:
        if save_pockets:
            self.pdb_writer = PDBWriter(chain_mapping)

        # Process receptor
        #pdb_file = PDBFile.read(self.receptor_path)
        #receptor_array = pdb_file.get_structure(model=1)
        receptor_data = self.process_receptor(
            receptor=receptor_array,
            receptor_path=self.receptor_path,
            chain_mapping=chain_mapping,
        )
        if not receptor_data:
            return None
        logger.info("Receptor processed.")
        # Process pocket
        pocket_data = self.extract_pocket(
            receptor=receptor_array,
            ligand_coords=ligand_data.coords, 
            chain_mapping=chain_mapping,
        )
        if not pocket_data:
            logger.warning(f"No pocket extracted for {self.receptor_path}")
            return None
        
        logger.info(f"Extracted pocket for {self.receptor_path}")

        if save_pockets:
                output_dir = os.path.dirname(self.receptor_path)
                pocket_path = os.path.join(output_dir, f"pocket.pdb")
                self.pdb_writer.write(pocket_data, pocket_path)

        if npnde_data:
                temp_npnde_data = npnde_data.copy()
        else:
                temp_npnde_data = {}

        system_data = SystemData(
            receptor=receptor_data, 
            ligand=ligand_data,
            pocket=pocket_data,
            pharmacophore=pharmacophore_data,
            system_id=None, 
            ligand_id=None, 
            npndes=temp_npnde_data,
        )
        return system_data
    #@profile
    def process_structures(
        self,
        ligand_mol: Chem.rdchem.Mol, #no longer a dictionary, 
        receptor_array: struc.AtomArray,
        npnde_mols: Optional[Dict[str, Chem.rdchem.Mol]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> Dict[str, Any]:
        # Process ligands, returns a LigandData object and PharmacophoreData object
        ligands_data, pharmacophore_data = self.process_ligand(
            ligand_mol=ligand_mol,
            receptor_array=receptor_array, 
            ligand_path=self.ligand_path,
        )

        npnde_data = {}
        if npnde_mols:
            npnde_data = self.process_npndes(npnde_mols)

        systems_data = self.process_structures_no_links(
            ligand_data=ligands_data,
            receptor_array=receptor_array,
            npnde_data=npnde_data,
            pharmacophore_data=pharmacophore_data,
            chain_mapping=chain_mapping,
            save_pockets=save_pockets,
        )
        
        if systems_data:
            return {
                "systems_list": systems_data,
                "links": False,
                #"annotation": self.system.system,
            }
        else:
            return None
    
    #adjust file paths if files end with .gninatypes
    def adjust_gninatypes_filepaths(self, receptor_path: Path, ligand_path: Path):
        # Adjust receptor file path
        if receptor_path.suffix == ".gninatypes":
            receptor_stem = receptor_path.stem

        if receptor_stem.endswith("_0"):
            receptor_stem = receptor_stem[:-2]  # remove _0
        
        receptor_path = receptor_path.with_name(f"{receptor_stem}.pdb")

        # Adjust ligand file path
        if ligand_path.suffix == ".gninatypes":
            ligand_stem = ligand_path.stem
        
        if ligand_stem.endswith("_0"):
            ligand_stem = ligand_stem[:-2]  # remove _0
        
        ligand_path = ligand_path.with_name(f"{ligand_stem}.sdf.gz")

        return receptor_path, ligand_path

    #when receptor file path ends with pdb and ligand with sdf
    # enforcing receptor files names to end with _rec.pdb
    # enforcing ligand files to end with either _docked.sdf.gz or _min.sdf.gz
    def adjust_pdb_sdf_filepaths(self, receptor_path: Path, ligand_path: Path):
        #Adjust receptor file path
        receptor_match = re.search(r'(.+_rec)', receptor_path.stem) #keep only up to _rec
        if receptor_match:
            receptor_stem = receptor_match.group(1)
            receptor_path = receptor_path.with_name(f"{receptor_stem}.pdb")
        else:
            logger.error(f"Unexpected receptor format: {receptor_path.stem}")

        #Adjust ligand file path
        ligand_stem = ligand_path.stem
        ligand_match = re.search(r'(.+?_(docked|min))', ligand_stem) # keep only up to _docked or _min

        if ligand_match:
            ligand_stem = ligand_match.group(1)
            ligand_path = ligand_path.with_name(f"{ligand_stem}.sdf.gz")
        else:
            logger.error(f"Ligand path does not contain _docked or _min: {ligand_stem}")

        return receptor_path, ligand_path
