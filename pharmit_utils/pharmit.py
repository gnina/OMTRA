import dgl
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import functools
import zarr
from collections import OrderedDict

from rdkit import Chem
from rdkit.Geometry import Point3D

from omtra.utils.zarr_utils import list_zarr_arrays
from omtra.utils.misc import classproperty
from omtra.constants import lig_atom_type_map, bond_type_map


class PharmitDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split: str,
                 return_type: str,
                 include_pharmacophore: bool = False,
                 include_extra_feats: bool = False,
                 n_chunks_cache: int = 4,
    ):
        super().__init__()

        zarr_store_path = Path(data_dir) / f'{split}.zarr'
        if not zarr_store_path.exists():
            raise ValueError(f"There is no zarr store at the path {zarr_store_path}")

        if return_type not in ['rdkit', 'dict']:
            return NotImplementedError("Returned molecule type must be 'rdkit' or 'dict'")

        self.store_path = zarr_store_path   
        self.store = zarr.storage.LocalStore(str(zarr_store_path), read_only=True)
        self.root = zarr.open(store=self.store, mode='r')
        self.n_chunks_cache = n_chunks_cache
        self.build_cached_chunk_fetchers()

        self.rows_per_chunk = {}
        for array_name in self.array_keys:
            self.rows_per_chunk[array_name] = self.root[array_name].chunks[0]

        self.return_type = return_type
        self.include_pharmacophore = include_pharmacophore
        self.include_extra_feats = include_extra_feats   

    @classproperty
    def name(cls):
        return 'pharmit'
    
    @property
    def n_zarr_chunks(self):
        return self.root['lig/node/x'].shape[0] // self.root['lig/node/x'].chunks[0]
    
    @property
    def graphs_per_chunk(self):
        return len(self) // self.n_zarr_chunks

    @functools.cached_property
    def array_keys(self):
        return list_zarr_arrays(self.root)
        
    def build_cached_chunk_fetchers(self):
        self.chunk_fetchers = {}
        for array_name in self.array_keys:
            self.chunk_fetchers[array_name] = ChunkFetcher(self.root, array_name, cache_size=self.n_chunks_cache)

    def __len__(self):
        return self.root['lig/node/graph_lookup'].shape[0]

    def slice_array(self, array_name, start_idx, end_idx=None):
        """Slice data from a zarr array but utilize chunk caching to minimize disk access."""

        if array_name not in self.array_keys:
            raise ValueError(f"There is no array with the name {array_name} in the zarr store located at {self.store_path}")

        single_idx = False
        if end_idx is None:
            single_idx = True
            end_idx = start_idx+1

        chunk_size = self.rows_per_chunk[array_name] # the number of rows in each chunk (we only chunk and slice along the first dimension)

        # get the chunk id, as well as the index of our slice relative to the chunk, for the start and end of the slice
        start_chunk_id, start_chunk_idx = divmod(start_idx, chunk_size)
        end_chunk_id, end_chunk_idx = divmod(end_idx, chunk_size)

        # retrieve all chunks that are "touched" by the slice, using the cached chunk fetchers
        chunks = [self.chunk_fetchers[array_name](chunk_id) for chunk_id in range(start_chunk_id, end_chunk_id + 1)]

        # slice just the data we want from the chunks
        if len(chunks) == 1:
            chunk_slices = [  chunks[0][start_chunk_idx:end_chunk_idx]  ]
        else:
            chunk_slices = []
            for i in range(len(chunks)):
                if i == 0:
                    chunk_slices = [chunks[i][start_chunk_idx:]]
                elif i == len(chunks) - 1:
                    chunk_slices.append(chunks[i][:end_chunk_idx])
                else:
                    chunk_slices.append(chunks[i])

        data = np.concatenate(chunk_slices)
        if single_idx:
            data = data.squeeze()
        return data

    def __getitem__(self, idx) -> dgl.DGLHeteroGraph:

        # slice lig node data
        data_dict = {}

        start_idx, end_idx = self.slice_array('lig/node/graph_lookup', idx)
        start_idx, end_idx = int(start_idx), int(end_idx)

        for nfeat in ['x', 'a', 'c']:
            data_dict[nfeat] = self.slice_array(f'lig/node/{nfeat}', start_idx, end_idx)

        if self.include_extra_feats and self.return_type == 'dict':
            # Get extra ligand atom features as a dictionary
            data_dict['extra_feats'] = {}

            extra_feats = self.slice_array(f'lig/node/extra_feats', start_idx, end_idx)
            extra_feats = extra_feats[:, :-1]
            features = self.root['lig/node/extra_feats'].attrs.get('features', [])

            # Iterate over all but the last feature
            for col_idx, feat in enumerate(features[:-1]):
                col_data = extra_feats[:, col_idx]
                if feat == 'chiral':
                    data_dict['extra_feats']['chiral_binary'] = torch.from_numpy(col_data).long()
                else:         
                    data_dict['extra_feats'][feat] = torch.from_numpy(col_data).long()
            
            chiral_data = self.slice_array(f'lig/node/chirality', start_idx, end_idx)[:,0]
            data_dict['extra_feats']['chiral'] = torch.from_numpy(chiral_data).long()
            
        # get slice indicies for ligand-ligand edges
        edge_slice_idxs = self.slice_array('lig/edge/graph_lookup', idx)

        # slice ligand-ligand edge data
        start_idx, end_idx = edge_slice_idxs
        start_idx, end_idx = int(start_idx), int(end_idx)
        data_dict['e'] = self.slice_array('lig/edge/e', start_idx, end_idx)
        data_dict['edge_idxs'] = self.slice_array('lig/edge/edge_index', start_idx, end_idx)

        # convert to torch tensors and set data types
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)
                if k == 'x':
                    data_dict[k] = data_dict[k].float()
                else:
                    data_dict[k] = data_dict[k].long()

        if self.return_type == 'rdkit':
            # Convert from integer encoding to true atom types and charges
            data_dict['a'] = [lig_atom_type_map[int(atom)] for atom in data_dict['a']]

            mol = self.build_rdkit_mol(data_dict)
            return mol
        
        elif self.return_type == 'dict':
            # if this task includes pharmacophore data, then we need to slice and add that data to the graph
            if self.include_pharmacophore:
                # read pharmacophore data from zarr store
                start_idx, end_idx = self.slice_array('pharm/node/graph_lookup', idx)
                data_dict['pharm'] = {}
                data_dict['pharm']['x'] = torch.from_numpy(self.slice_array('pharm/node/x', start_idx, end_idx)).float()
                data_dict['pharm']['v'] = torch.from_numpy(self.slice_array('pharm/node/v', start_idx, end_idx)).long()
                data_dict['pharm']['a'] = torch.from_numpy(self.slice_array('pharm/node/a', start_idx, end_idx)).long()

            mol = data_dict

        return mol
    
    def retrieve_atom_idxs(self, idx) -> tuple:
        start_idx, end_idx = self.slice_array('lig/node/graph_lookup', idx)
        return start_idx, end_idx
    
    def retrieve_edge_idxs(self, idx) -> tuple:
        start_idx, end_idx = self.slice_array('lig/edge/graph_lookup', idx)
        return start_idx, end_idx

    def build_rdkit_mol(self, data_dict):
        """Builds a rdkit molecule from the given atom and bond information."""
        
        positions = data_dict['x']
        atom_types = data_dict['a']
        atom_charges = data_dict['c']
        bond_types = data_dict['e']
        bond_src_idxs = data_dict['edge_idxs'][:, 0]
        bond_dst_idxs = data_dict['edge_idxs'][:, 1]
        
        # create an rdkit molecule and add atoms to it
        mol = Chem.RWMol()
        for atom_type, charge in zip(atom_types, atom_charges):
            a = Chem.Atom(atom_type)
            if charge != 0:
                a.SetFormalCharge(int(charge))
            mol.AddAtom(a)

        # add bonds to rdkit molecule
        for bond_type, src_idx, dst_idx in zip(
            bond_types, bond_src_idxs, bond_dst_idxs
        ):
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
            mol.AddBond(src_idx, dst_idx, bond_type_map[bond_type])

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Warning: Molecule failed to Kekulize.")
            return None

        # Set coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x, y, z = positions[i]
            x, y, z = float(x), float(y), float(z)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf)

        return mol

class ChunkFetcher:
    def __init__(self, root, array_name, cache_size):
        self.root = root
        self.array_name = array_name
        self.array = self.root[self.array_name]
        self.cache_size = cache_size
        self.cache = OrderedDict()  # Ordered dictionary to maintain LRU order
        self.chunk_size = self.array.chunks[0]

    def __call__(self, chunk_id):
        if chunk_id in self.cache:
            # Move the accessed chunk to the end to mark it as recently used
            self.cache.move_to_end(chunk_id)
        else:
            if len(self.cache) >= self.cache_size:
                # Remove the least recently used chunk
                self.cache.popitem(last=False)

            # Fetch the chunk and add it to the cache
            chunk_start_idx = chunk_id * self.chunk_size
            chunk_end_idx = chunk_start_idx + self.chunk_size
            self.cache[chunk_id] = self.array[chunk_start_idx:chunk_end_idx]

        return self.cache[chunk_id]