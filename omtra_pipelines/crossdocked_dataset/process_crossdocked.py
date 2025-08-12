#Import libraries
import re, subprocess, os, gzip, json, glob, multiprocessing
import numpy as np
from rdkit.Chem import AllChem as Chem
import tempfile
import pickle
from tqdm import tqdm
#from Bio.PDB import PDBParser
#from Bio.PDB.Polypeptide import is_aa
from scipy.spatial.distance import cdist
#import Bio
#import Bio.SeqUtils
import argparse
from pathlib import Path
import yaml
from functools import partial
#from pharmacoforge.constants import ph_type_to_idx
from tqdm.contrib.concurrent import process_map
from typing import Dict
#from pharmacoforge.dataset.receptor_utils import get_mol_pharm
# !!: Added
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure import filter_amino_acids, get_residue_masks
from biotite.structure import concat
# Disable RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Parse command line arguments
# This function parses command line arguments using argparse
# 3 arguments are expected: directory to crossdocked data, path to config file, and the number of workers for multiprocessing
def parse_args():
    parser = argparse.ArgumentParser()
    #added line for path to crossdocked dataset
    parser.add_argument("--directory", help="Path to the directory containing the crossdocked dataset", required=True, type=Path)
    parser.add_argument("--config", help="Path to config file", required=True, type=Path)
    parser.add_argument('--max_workers', type=int, default=None, help='Number of workers to use for multiprocessing, default to max available.')
    args = parser.parse_args()
    return args

# Element fixer function converts an element string to a standard format
# leaves the first character of the element as is and converts the rest to lowercase
def element_fixer(element: str):
    if len(element) > 1:
        element = element[0] + element[1:].lower()
    return element

# Get features (!!: assuming reclig will be a tuple of the receptor and ligand file names as stored in 1 line of a types file)
# Input: reclig (a tuple of length 2, where first entry is receptor file name and second entry is ligand file name)
def getfeatures(reclig, crossdocked_data_dir: Path, pocket_cutoff: int = 8):
    rec, glig = reclig # unpack tuple into receptor & ligand file names
    
    ####### Prepare receptor and ligand file names #####

    rec = rec.replace('_0.gninatypes','.pdb')
    
    m = re.search('(\S+)_(\d+)\.gninatypes',glig) #might need r before the string
    prefix = m.group(1) #part of the filename before the first underscore _
    num = int(m.group(2)) #part of the filename after the first underscore _ which is a number
    lig = prefix+'.sdf.gz' #final ligand file name
    
    rec_path = crossdocked_data_dir / rec
    lig_path = crossdocked_data_dir / lig
    rec_path = str(rec_path)
    lig_path = str(lig_path)

    ####### Print some warnings if path does not exist #########
    if not os.path.exists(rec_path):
        print(f"Warning: {rec_path} does not exist")
    if not os.path.exists(lig_path):
        print(f"Warning: {lig_path} does not exist")
    

    with tempfile.TemporaryDirectory() as tmp: #create a temp folder to hold temp lig and rec files
        try:
            # Extract one ligand conformer/pose from compressed sdf file and save it to a temp file for processing
            if num != 0:
                #extract num conformer if ligand is not 0th conformer
                sdf = gzip.open(lig_path).read().split(b'$$$$\n')[num]+b'$$$$\n'
                lig_path = os.path.join(tmp,'lig.sdf') #create temp lig file
                with open(lig_path, 'wb') as out: #open temp lig file in write mode
                    out.write(sdf)

            ##### Extract pharmacophore pocket & Get Ligand Coordinates ##############

            # !!: extract residue/atom-level pocket using biotite for receptor
            #pdb_struct = load_structure(rec_path) #biotite AtomArray object (gives chain, coordinates, bonds, etc.)
            pdb_file = PDBFile.read(rec_path)
            pdb_struct = pdb_file.get_structure(model=1) #!!: check if this is correct
            #use rdkit to get extract atom-level ligand coordinates
            if lig_path.endswith('.gz'):
                with gzip.open(lig_path) as f:
                    supp = Chem.ForwardSDMolSupplier(f,sanitize=False)
                    ligand = next(supp)
                    del supp
                
            else:
                supp = Chem.ForwardSDMolSupplier(lig_path,sanitize=False)
                ligand = next(supp)
                del supp
            lig_coords = ligand.GetConformer().GetPositions()
            
            ###### Get Atom-level residues from receptor side that close to ligand #########
            # !!: use biotite to get residues which constitute the binding pocket
            # !!: i think we make a dist matrix between all atoms in the receptor and all atoms in the ligand (before we cared about feature type, here we dont need to)
            # We are keeping non standard residues now
            residue_masks = get_residue_masks(pdb_struct) # get masks for each residue in receptor (boolean array of size (N,1) giving whether atom n belongs to residue i)
            amino_mask = filter_amino_acids(pdb_struct) # filter for non-amino acids (1D boolean array indicating of atom n is amino acid)
            pocket_residues = []

            #loop through each residue mask (i is the index of the residue, mask is boolean array of that residue)
            for i, mask in enumerate(residue_masks):
                residue_atoms = pdb_struct[mask] # get all atoms for a residue

                #skip nonstandard residues
                if not np.any(amino_mask[mask]):
                    continue
                
                res_coords = residue_atoms.coord # get coordinates of the residue

                # if any atom in the residue is within the cutoff distance from the ligand, add it to the pocket residues
                min_rl_dist = cdist(res_coords, lig_coords).min() # compute distance between ligand and residue
                if min_rl_dist < pocket_cutoff:
                    pocket_residues.append(residue_atoms)
           
           
            #dist_matrix = cdist(pdb_struct.coord, lig_coords)
            #close_atoms = np.any(dist_matrix <= pocket_cutoff, axis=1)
            #pocket_residues = pdb_struct[close_atoms] #!!: an AtomArray with only receptor atoms close to ligand

            ###### Prepare Output Pocket Information ########
            # filter our H atoms from the pocket residues
            #pocket_atoms = pocket_residues[pocket_residues.element != 'H'] #remove hydrogens from receptor
            #flatten into single AtomArray
            flat_res = concat(pocket_residues)
            pocket_atoms = flat_res[flat_res.element != "H"]
            pocket_coords = pocket_atoms.coord #get coordinates of receptor atoms
            pocket_elements = np.array([element_fixer(a) for a in pocket_atoms.element]) #get elements of receptor atoms
            pocket_anames = pocket_atoms.atom_name #get atom names of receptor atoms
            #pocket_res = np.array([Bio.PDB.Polypeptide.three_to_index(name) for name in pocket_atoms.res_name]) #get 3-letter code of receptor atoms
            pocket_rid = pocket_atoms.res_id #get residue ids of receptor atoms

            #return ((rec,glig,ligand,(feature_coords, feature_kind),(pocket_coords, pocket_elements, pocket_feat_coords, pocket_feat_kind, pocket_anames, pocket_res, pocket_rid)))
            return (rec, glig, ligand, pdb_struct, (pocket_coords, pocket_elements, pocket_anames, pocket_rid))

        except Exception as e:
            print(e)
            print(rec,glig)
            return((rec,glig,None,None,None))

#Write processed dataset processes into tensors
# !!: data is the return value of getfeatures function
def write_processed_dataset(processed_data_dir: str, types_file_path: str, data: list, pocket_element_map: list, min_pharm_centers = 3):
    # Map each atom element to an index
    pocket_element_to_idx = {element: idx for idx, element in enumerate(pocket_element_map)}
    prot_file_name = []
    pharm_file_name = [] #!!: should we call this lig_file_name?
    rec_struct = [] #added to store receptor structure
    lig_rdmol = []
    prot_pos_arr = []
    prot_feat_arr = []

    # Loop through the data (returned from getfeatures) and extract the relevant information
    for item in data:

        prot_file_name.append(item[0]) # receptor file name
        pharm_file_name.append(item[1]) # ligand file name
        lig_rdmol.append(item[2]) # ligand rdkit molecule
        rec_struct.append(item[3]) #added to store receptor structure

        prot_pos_arr.append(item[4][0]) #pocket coordinates
        prot_feat_arr.append(item[4][1]) #pocket elements

    # get the number of receptor atoms in every example
    n_atoms = np.array([len(x) for x in prot_pos_arr])

    # concatenate prot_pos (pocket coordinates) into single arrays
    prot_pos = np.concatenate(prot_pos_arr, axis=0, dtype=np.float32)

    # convert pocket elements from strings to integers and concatenate into a single array
    prot_feat = np.concatenate(prot_feat_arr, axis=0) #pocket elements
    prot_feat_idxs = np.array([pocket_element_to_idx[el] for el in prot_feat]) #convert pocket elements to indices
    prot_feat = np.array(prot_feat_idxs, dtype=np.int32)

    # create an array of indicies to keep track of the start_idx and end_idx of receptor atoms for each protein-ligand pair
    # example: [0, 10], [10, 20], [20, 30] for 3 proteins with 10 atoms each
    prot_idx_array = np.zeros((len(prot_pos_arr), 2), dtype=int)
    prot_idx_array[:, 1] = np.cumsum(n_atoms)
    prot_idx_array[1:, 0] = prot_idx_array[:-1, 1]
    
    # get the processed output directory for this types file
    types_file_stem = Path(types_file_path).name.split('.types')[0]
    output_dir = Path(processed_data_dir) / types_file_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # write the ligand rdkit molecules to a .pkl.gz file
    lig_rdmol_file = output_dir / 'lig_rdmol.pkl.gz'
    with gzip.open(lig_rdmol_file, 'wb') as f:
        pickle.dump(lig_rdmol, f)

    # write the protein file names to a .pkl.gz file
    prot_file_name_file = output_dir / 'prot_file_names.pkl.gz'
    with gzip.open(prot_file_name_file, 'wb') as f:
        pickle.dump(prot_file_name, f)
    

if __name__ == "__main__":

    args = parse_args()

    # process config file into dictionary
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    crossdocked_path = config['dataset']['raw_data_dir']
    crossdocked_data_dir = Path(crossdocked_path) / 'CrossDocked2020'
    output_path = config['dataset']['processed_data_dir']
    dataset_size = config['dataset']['dataset_size']

    # all inputs is a list of tuples. Each tuple has length 2.
    # the first entry in the tuple is the filepath of the types file for which this data point came from
    # the second entry in the tuple is itself a list of tuples. Each tuple has length 2.
    # the first entry in the tuple is the filepath of the receptor file, the second entry is the filepath of the ligand file
    allinputs = []
    types_files = os.path.join(crossdocked_path,'types','it2_tt_v1.3_0_test*types')
    for fname in glob.glob(types_files):
        #pull out good rmsd lines only
        f = open(fname)
        # inputs is a list which contains tuples of length 2. the first item in each tuple is the receptor file name and the second item is the ligand file name
        inputs = [] 
        for idx, line in enumerate(f):
            label,affinity,rmsd,rec,glig,_ = line.split()
            if label == '1':
                inputs.append((rec,glig))

            if dataset_size is not None and idx > dataset_size:
                break

        allinputs.append((fname,inputs))
    
    #set the arguments from config which need to be passed to the getfeatures function
    # uses partial to prefill some arguments into get features function
    getfeatures_partial = partial(getfeatures, crossdocked_data_dir=crossdocked_data_dir, pocket_cutoff=config['dataset']['pocket_cutoff'])

    allphdata = []
    for fname, inputs in allinputs:
        
        # if args.max_workers is None:
        #     num_workers = multiprocessing.cpu_count()
        # else:
        #     num_workers = args.max_workers
        
        # chunksize = len(inputs) // num_workers
        chunksize = 20

        # print the fname we are processing
        print(f'processing types file {fname}')

        # extract features for each protein-ligand pair
        if args.max_workers:
            phdata = process_map(getfeatures_partial, inputs, max_workers=args.max_workers, chunksize=chunksize)
        else:
            phdata = process_map(getfeatures_partial, inputs, chunksize=chunksize)

        n_samples = len(phdata)

        # the third entry in each tuple is the ligand molecule as an rdkit object, which is None if 
        # the ligand molecule could not be parsed. we filter out these examples
        phdata = [ex for ex in phdata if ex[2]]

        n_bad_ligands = n_samples - len(phdata)

        print(f'{n_samples} samples in {fname}')
        #print(f'failed to parse {n_bad_ligands} ligands and failed to obtain pharmacophore points for {n_bad_pharm} examples')
        print(f'processed {len(phdata)} examples')
        
        # process into tensors
        write_processed_dataset(output_path, fname, phdata,
                                pocket_element_map=config['dataset']['prot_elements'],
                                min_pharm_centers=config['dataset']['min_pharm_centers'])














            
            





            

            












