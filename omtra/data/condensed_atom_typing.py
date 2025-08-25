import numpy as np
import pickle
import torch

from omtra.utils import omtra_root
from omtra.constants import lig_atom_type_map, charge_map, extra_feats_map


class CondensedAtomTyper():
    def __init__(self,
                 fake_atoms: bool):
        
        # load tuple counts for Pharmit
        cond_a_path = omtra_root() + '/omtra/constants/pharmit_condensed_atom_types.pkl'
        
        with open(cond_a_path, 'rb') as f:
            cond_a_counts = pickle.load(f)    
        
        # convert list of unique tuples into a numpy array, cond_to_uncond
        cond_a_list = list(cond_a_counts.keys())
        cond_to_uncond = np.array(cond_a_list, dtype=np.int64)  # array of shape (n_types, n_feats)

        # load tuple counts for Plinder
        plinder_cond_a_path = omtra_root() + '/omtra/constants/plinder_condensed_atom_types.pkl'
        with open(plinder_cond_a_path, 'rb') as f:
            plinder_cond_a_counts = pickle.load(f)    

        # unpack unique tuples from each version of the plinder dataset
        plinder_uncond_tuples = []
        for plinder_version_cond_a in plinder_cond_a_counts.values():
            plinder_uncond_tuples.append(np.array(list(plinder_version_cond_a.keys()), dtype=np.int64))

        # Concatenate arrays and find unique atom uncondensed feature tuples
        all_tuples = np.vstack([cond_to_uncond] + plinder_uncond_tuples)
        cond_to_uncond = np.unique(all_tuples, axis=0)
        self.cond_to_uncond = cond_to_uncond

        self.n_real_categories = self.cond_to_uncond.shape[0]

    
        self.fake_atoms = fake_atoms
        self.lig_feats = ['a', 'c'] + list(extra_feats_map.keys())

        if self.fake_atoms:
            self.fake_atom_tuple = (len(lig_atom_type_map),) + (0,) * (len(self.lig_feats)- 1)
            self.masked_atom_tuple = (len(lig_atom_type_map)+1, len(charge_map), ) + tuple(extra_feats_map.values())
            self.fake_atom_idx = self.cond_to_uncond.shape[0]
            self.masked_atom_idx = self.cond_to_uncond.shape[0] + 1

            extra_uncond_tuples = [self.fake_atom_tuple, self.masked_atom_tuple]
            
        else:
            self.fake_atom_tuple = None
            self.masked_atom_tuple = (len(lig_atom_type_map), len(charge_map), ) + tuple(extra_feats_map.values())
            self.fake_atom_idx = None
            self.masked_atom_idx = self.cond_to_uncond.shape[0]

            extra_uncond_tuples = [self.masked_atom_tuple]

        extra_uncond_tuples = np.array(extra_uncond_tuples, dtype=int)
        self.cond_to_uncond = np.concatenate([self.cond_to_uncond] + [extra_uncond_tuples], axis=0)


        # create a dictionary for rapid converison of uncondensed type tuples -> condensed type idxs
        self.uncond_tuple_to_cond_idx = {
            tuple(self.cond_to_uncond[i].tolist()): i
            for i in range(self.cond_to_uncond.shape[0])
        }

    def feats_to_cond_a(self,
                          a: np.array,
                          c: np.array,
                          extra_feats: np.array
                          ):
        """Convert individual atom features (atom type, charge, and extra_feats such as hybridization, implicit hydrogens, etc.) to condensed atom type indices."""
        input_uncond_feats = np.concatenate([a[:, None], c[:, None], extra_feats], axis=1)

        unique_uncond_feats, inverse = np.unique(input_uncond_feats, axis=0, return_inverse=True)

        unique_cond_feats = []
        for uf in unique_uncond_feats:
            uf = tuple(uf)
            if uf in self.uncond_tuple_to_cond_idx:
                unique_cond_feats.append(self.uncond_tuple_to_cond_idx[uf])
            else:
                raise ValueError(f"Encountered invalid atom feature tuple: {uf}")

        cond_feats = np.array(unique_cond_feats)[inverse]
        return cond_feats
    
    def cond_a_to_feats(self,
                        cond_a: np.array):
        """Convert condensed atom type indices back to individual atom features (atom type, charge, and extra_feats such as hybridization, implicit hydrogens, etc.)."""

        # map condensed atom types to an array of uncondensed atom features
        try:
            uncond_arr = self.cond_to_uncond[cond_a]
        except IndexError:
            raise ValueError(f"Encountered invalid condensed atom index")

        
        # recover atom features as separate tensors
        lig_feat_dict = {}

        for i, feat in enumerate(self.lig_feats):
            lig_feat_dict[feat] = uncond_arr[:, i]
        
        # convert charges to token indicies
        charge_map_array = np.array(charge_map)
        lig_feat_dict['c'] = np.searchsorted(charge_map_array, lig_feat_dict['c'])

        return lig_feat_dict