from pathlib import Path
from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.plinder import LigandData
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    bond_type_map,
    charge_map,
    protein_element_map,
    protein_atom_map,
    residue_map,
)

import warnings

# Suppress the specific warning from vlen_utf8.py
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    module="zarr.codecs.vlen_utf8",
)

class PlinderLigandExtractor(ZarrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])


    def __getitem__(self, idx):
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == idx
        ].iloc[0]

        lig_atom_start, lig_atom_end = (
            int(system_info["lig_atom_start"]),
            int(system_info["lig_atom_end"]),
        )
        lig_bond_start, lig_bond_end = (
            int(system_info["lig_bond_start"]),
            int(system_info["lig_bond_end"]),
        )


        ligand = LigandData(
            sdf=system_info["lig_sdf"],
            ccd=system_info["ccd"],
            is_covalent=False,
            coords=self.slice_array("ligand/coords", lig_atom_start, lig_atom_end),  # x
            atom_types=self.slice_array(
                "ligand/atom_types", lig_atom_start, lig_atom_end
            ),  # a
            atom_charges=self.slice_array(
                "ligand/atom_charges", lig_atom_start, lig_atom_end
            ),  # c
            bond_types=self.slice_array(
                "ligand/bond_types", lig_bond_start, lig_bond_end
            ),  # e
            bond_indices=self.slice_array(
                "ligand/bond_indices", lig_bond_start, lig_bond_end
            ),  # edge index
        )

        rdkit_mol = self.to_rdkit_molecule(ligand)
        fp = Chem.RDKFingerprint(rdkit_mol)
        smiles = Chem.MolToSmiles(rdkit_mol)
        return dict(fp=fp, smiles=smiles)

    def to_rdkit_molecule(self, ligand: LigandData):
        """Builds a rdkit molecule from the given atom and bond information."""
        # create a rdkit molecule and add atoms to it

        atom_types = [ lig_atom_type_map[i] for i in ligand.atom_types.tolist() ]
        atom_charges = ligand.atom_charges.tolist()
        bond_src_idxs, bond_dst_idxs = ligand.bond_indices.T
        bond_types = ligand.bond_types.tolist()
        positions = ligand.coords

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
            return None

        # Set coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x, y, z = positions[i]
            x, y, z = float(x), float(y), float(z)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf)

        return mol

    def __len__(self):
        return self.system_lookup.shape[0]
    
    def get_num_nodes(self, *args, **kwargs):
        pass

    def name(self):
        return "plinder_ligand_extractor"
    
    def retrieve_graph_chunks(self):
        pass