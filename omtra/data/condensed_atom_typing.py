import numpy as np
import pickle

from omtra.utils import omtra_root
from omtra.constants import lig_atom_type_map


class CondensedAtomTyper():
    def __init__(self,
                 fake_atoms: bool):
        cond_a_path = omtra_root() + '/omtra/constants/pharmit_condensed_atom_types.pkl'
        
        with open(cond_a_path, 'rb') as f:
            cond_a_counts = pickle.load(f)    
        
        self.cond_a_list = sorted(list(cond_a_counts.keys()))
        self.fake_atoms = fake_atoms
        self.lig_feats = ['a', 'c', 'impl_H', 'aro', 'hyb', 'ring', 'chiral']

        if self.fake_atoms:
            self.fake_atom_tuple = (len(lig_atom_type_map),) + (0,) * (len(self.cond_a_list[0])- 1)
            self.masked_atom_tuple = (len(lig_atom_type_map)+1,) + (0,) * (len(self.cond_a_list[0])- 1)
        else:
            self.fake_atom_tuple = None
            self.masked_atom_tuple = (len(lig_atom_type_map),) + (0,) * (len(self.cond_a_list[0])- 1)

        
    def feats_to_cond_a(self,
                          a: np.array,
                          c: np.array,
                          extra_feats: np.array
                          ):
        # Convert from atom features to condensed atom type index

        lig_feat_tuples =  [(atom_type, atom_charge, *ef) for atom_type, atom_charge, ef in zip(a.tolist(), c.tolist(), extra_feats.tolist())]

        cond_a = []

        for lig_feat_tuple in lig_feat_tuples:
            try:
                if (self.fake_atom_tuple is not None) and (lig_feat_tuple == self.fake_atom_tuple):  # fake atom type
                    cond_a.append(len(self.cond_a_list))

                elif lig_feat_tuple == self.masked_atom_tuple:  # masked atom type
                    cond_a.append(len(self.cond_a_list)+1)
                
                else:
                    cond_a.append(self.cond_a_list.index(lig_feat_tuple))

            except Exception as e:
                print(f"Encountered invalid atom feature tuple: {lig_feat_tuple}")

        cond_a = np.array(cond_a)
        
        return cond_a
    
    def cond_a_to_feats(self,
                        cond_a: np.array):
        # Convert from condensed atom type representation to explicit atom features

        lig_feat_tuples = []

        for idx in cond_a:
            try:
                if idx == len(self.cond_a_list):    
                    if self.fake_atoms:
                        lig_feat_tuples.append(self.fake_atom_tuple)    # fake atom
                    else:
                        lig_feat_tuples.append(self.masked_atom_tuple)  # no fake atoms, this is a masked atom

                elif idx == (len(self.cond_a_list)+1):  # can only be masked atom
                    lig_feat_tuples.append(self.masked_atom_tuple)
                else:
                    lig_feat_tuples.append(self.cond_a_list[idx])
            except Exception as e:
                print(f"Warning: Encountered an invalid condensed atom type {idx}")
       
        lig_feat_tuples = np.array(lig_feat_tuples)
        
        # recover atom features as separate tensors
        lig_feat_dict = {}

        for i, feat in enumerate(self.lig_feats):
            lig_feat_dict[feat] = lig_feat_tuples[:, i]
        
        return lig_feat_dict