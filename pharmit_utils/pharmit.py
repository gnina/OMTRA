import dgl
import torch
import numpy as np
from omegaconf import DictConfig
import math

from rdkit import Chem
from rdkit.Geometry import Point3D

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.utils.misc import classproperty
from omtra.constants import lig_atom_type_map, charge_map, bond_type_map, ph_idx_to_type


class PharmitDataset(ZarrDataset):
    def __init__(self, 
                 processed_data_dir: str,
                 split: str,
                 return_type: str,
                 include_pharmacophore: bool = False,
                 include_extra_feats: bool = False,
    ):
        super().__init__(split, processed_data_dir)

        if return_type not in ['rdkit', 'dict']:
            return NotImplementedError("Returned molecule type must be 'rdkit' or 'dict'")
        
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


    def __len__(self):
        return self.root['lig/node/graph_lookup'].shape[0]

    def __getitem__(self, idx) -> dgl.DGLHeteroGraph:

        # slice lig node data
        data_dict = {}

        start_idx, end_idx = self.slice_array('lig/node/graph_lookup', idx)
        start_idx, end_idx = int(start_idx), int(end_idx)

        for nfeat in ['x', 'a', 'c']:
            data_dict[nfeat] = self.slice_array(f'lig/node/{nfeat}', start_idx, end_idx)

        if self.include_extra_feats:
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

        # TODO: add chiral edge types

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
                #data_dict['pharm']['v'] = torch.from_numpy(self.slice_array('pharm/node/v', start_idx, end_idx)).long()
                data_dict['pharm']['a'] = torch.from_numpy(self.slice_array('pharm/node/a', start_idx, end_idx)).long()

            mol = data_dict

        return mol
    
    def retrieve_atom_idxs(self, idx) -> tuple:
        start_idx, end_idx = self.slice_array('lig/node/graph_lookup', idx)
        return start_idx, end_idx
    
    def retrieve_edge_idxs(self, idx) -> tuple:
        start_idx, end_idx = self.slice_array('lig/edge/graph_lookup', idx)
        return start_idx, end_idx

    def retrieve_graph_chunks(self, frac_start: float, frac_end: float):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk. 
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self)
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        # construct a tensor containing the index ranges for each chunk
        chunk_index = torch.zeros(n_chunks, 2, dtype=torch.int64)
        chunk_index[:, 0] = self.graphs_per_chunk*torch.arange(n_chunks)
        chunk_index[:-1, 1] = chunk_index[1:, 0]
        chunk_index[-1, 1] = n_graphs

        # if we need to only expose a subset of chunks (due to distributed training), do so here
        if not (frac_start == 0.0 and frac_end == 1.0):
            start_chunk_idx = math.floor(frac_start * n_chunks)
            end_chunk_idx = math.floor(frac_end * n_chunks)
            chunk_index = chunk_index[start_chunk_idx:end_chunk_idx]

        return chunk_index
    
    def get_num_nodes(self, start_idx, end_idx, per_ntype=False):
        # TODO: I don't know if we want this anymore
        # here, unlike in other places, start_idx and end_idx are 
        # indexes into the graph_lookup array, not a node/edge data array

        node_types = ['lig']
        if self.include_pharmacophore:
            node_types.append('pharm')

        node_counts = []
        for ntype in node_types:
            graph_lookup = self.slice_array(f'{ntype}/node/graph_lookup', start_idx, end_idx)
            ntype_node_counts = graph_lookup[:, 1] - graph_lookup[:, 0]
            node_counts.append(ntype_node_counts)

        if per_ntype:
            num_nodes_dict = {ntype: ncount for ntype, ncount in zip(node_types, node_counts)}
            return num_nodes_dict

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts

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

        # add extra features as atom properties
        if self.include_extra_feats:
            for a in mol.GetAtoms():
                for k, v in data_dict['extra_feats'].items():
                    feat = v[a.GetIdx()]
                    a.SetProp(k, str(feat))

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