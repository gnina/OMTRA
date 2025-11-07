import dgl
import torch
import math
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.constants import (
    lig_atom_type_map,
    charge_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    aa_substitutions,
    residue_map,
    protein_element_map,
    protein_atom_map,
)
from omtra.utils.rotation import rotate_ground_truth, center_on_ligand_gt, system_offset
from omtra.data.graph import build_complex_graph
from omtra.data.graph import edge_builders, approx_n_edges
from omtra.data.xace_ligand import add_k_hop_edges, MolXACE, add_fake_atoms
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty
from omtra.priors.prior_factory import get_prior
from omtra.priors.sample import sample_priors
from omtra.tasks.modalities import name_to_modality
from omtra.data.plinder import (
    LigandData,
    PharmacophoreData,
    StructureData,
    SystemData,
    BackboneData,
)
from omtra.utils.embedding import residue_sinusoidal_encoding
from omtra.data.condensed_atom_typing import CondensedAtomTyper
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import biotite.structure as struc
from omtra.constants import DEFAULT_DISTANCE_RANGE
import functools
from scipy.spatial.distance import cdist

import warnings

# Suppress the specific warning from vlen_utf8.py
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    module="zarr.codecs.vlen_utf8",
)


class PlinderDataset(ZarrDataset):
    def __init__(
        self,
        link_version: str,
        split: str,
        processed_data_dir: str,
        graph_config: Optional[DictConfig] = None,
        prior_config: Optional[DictConfig] = None,
        fake_atom_p: float = 0.0,
        pskip_factor: float = 0.0, 
        # this is a parmaeter that controls whether/how we do weighted sampling of the the dataset
        # if pskip_factor = 1, we do uniform sampling over all clusters in the system, and if it is 0, we apply no weighted sampling.
        res_id_embed_dim: int = 64,
        max_pharms_sampled: int = 8,
        sys_offset_std: float = 0.0
    ):
        super().__init__(
            split,
            f"{processed_data_dir}/{link_version}"
            if link_version
            else f"{processed_data_dir}/no_links",
        )
        self.split = split
        self.link_version = link_version
        self.graph_config = graph_config
        self.prior_config = prior_config
        self.fake_atom_p = fake_atom_p
        self.use_fake_atoms = self.fake_atom_p > 0
        self.pskip_factor = pskip_factor
        self.weighted_sampling = pskip_factor > 0.0 and split == 'train'
        self.sys_offset_std = sys_offset_std

        self.res_id_embed_dim = res_id_embed_dim

        self.max_pharms_sampled = max_pharms_sampled

        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])
        self.npnde_lookup = pd.DataFrame(self.root.attrs["npnde_lookup"])

        if self.weighted_sampling:
            if 'cluster_id' not in self.system_lookup.columns:
                raise ValueError(f'Weighted sampling enabled but no cluster assignments found for plinder link version {link_version} and split {split}')
            self.system_lookup = compute_pskip(self.system_lookup, pskip_factor=self.pskip_factor)

        self.encode_element = {
            element: i for i, element in enumerate(protein_element_map)
        }
        self.encode_residue = {res: i for i, res in enumerate(residue_map)}
        self.encode_atom = {atom: i for i, atom in enumerate(protein_atom_map)}
        self.charge_map_tensor = torch.tensor(charge_map)

    @classproperty
    def name(cls):
        return "plinder"

    @property
    def n_zarr_chunks(self):
        coords_arr = self.root["pocket/coords"]
        n_atoms = coords_arr.shape[0]
        n_chunks = math.ceil(n_atoms / coords_arr.chunks[0])
        return n_chunks

    @property
    def graphs_per_chunk(self):
        return len(self) // self.n_zarr_chunks

    def __len__(self):
        return self.system_lookup.shape[0]

    # TODO: address LigandData change
    def get_npndes(self, npnde_idxs: List[int]) -> Dict[str, LigandData]:
        npndes = {}
        for idx in npnde_idxs:
            npnde_info = self.npnde_lookup[self.npnde_lookup["npnde_idx"] == idx].iloc[
                0
            ]

            key = npnde_info["npnde_id"]

            atom_start, atom_end = npnde_info["atom_start"], npnde_info["atom_end"]

            bond_start = (
                npnde_info["bond_start"]
                if not pd.isna(npnde_info["bond_start"])
                else None
            )
            bond_end = (
                npnde_info["bond_end"] if not pd.isna(npnde_info["bond_end"]) else None
            )

            is_covalent = False
            if npnde_info["linkages"]:
                is_covalent = True

            npndes[key] = LigandData(
                sdf=npnde_info["sdf"],
                ccd=npnde_info["ccd"],
                is_covalent=is_covalent,
                linkages=npnde_info["linkages"],
                coords=self.slice_array("npnde/coords", atom_start, atom_end),
                atom_types=self.slice_array("npnde/atom_types", atom_start, atom_end),
                atom_charges=self.slice_array(
                    "npnde/atom_charges", atom_start, atom_end
                ),
                bond_types=self.slice_array(
                    "npnde/bond_types", int(bond_start), int(bond_end)
                )
                if bond_start is not None and bond_end is not None
                else np.zeros((0,), dtype=np.int32),
                bond_indices=self.slice_array(
                    "npnde/bond_indices", int(bond_start), int(bond_end)
                )
                if bond_start is not None and bond_end is not None
                else np.zeros((0, 2), dtype=np.int32),
            )
        return npndes

    @functools.lru_cache()
    def get_condensed_atom_typer(self):
        return CondensedAtomTyper(fake_atoms=self.fake_atom_p>0.0)
    
    def get_system(self,
                   index: int, 
                   include_pharmacophore: bool, 
                   include_protein: bool, 
                   include_extra_feats: bool, 
                   condensed_atom_typing: bool
    ) -> SystemData:
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == index
        ].iloc[0]

        rec_start, rec_end = int(system_info["rec_start"]), int(system_info["rec_end"])
        backbone_start, backbone_end = (
            int(system_info["backbone_start"]),
            int(system_info["backbone_end"]),
        )

        lig_atom_start, lig_atom_end = (
            int(system_info["lig_atom_start"]),
            int(system_info["lig_atom_end"]),
        )
        lig_bond_start, lig_bond_end = (
            int(system_info["lig_bond_start"]),
            int(system_info["lig_bond_end"]),
        )

        pocket_start, pocket_end = (
            int(system_info["pocket_start"]),
            int(system_info["pocket_end"]),
        )
        pocket_bb_start, pocket_bb_end = (
            int(system_info["pocket_bb_start"]),
            int(system_info["pocket_bb_end"]),
        )

        link_type = system_info["link_type"]
        if link_type:
            link_start, link_end = (
                int(system_info["link_start"]),
                int(system_info["link_end"]),
            )
            link_bb_start, link_bb_end = (
                int(system_info["link_bb_start"]),
                int(system_info["link_bb_end"]),
            )


        if include_protein:
            backbone = BackboneData(
                coords=self.slice_array(
                    "receptor/backbone_coords", backbone_start, backbone_end
                ),
                res_ids=self.slice_array(
                    "receptor/backbone_res_ids", backbone_start, backbone_end
                ),
                res_names=self.slice_array(
                    "receptor/backbone_res_names", backbone_start, backbone_end
                ),
                chain_ids=self.slice_array(
                    "receptor/backbone_chain_ids", backbone_start, backbone_end
                ),
            )

            receptor = StructureData(
                coords=self.slice_array("receptor/coords", rec_start, rec_end),
                atom_names=self.slice_array("receptor/atom_names", rec_start, rec_end),
                elements=self.slice_array("receptor/elements", rec_start, rec_end),
                res_ids=self.slice_array("receptor/res_ids", rec_start, rec_end),
                res_names=self.slice_array("receptor/res_names", rec_start, rec_end),
                chain_ids=self.slice_array("receptor/chain_ids", rec_start, rec_end),
                backbone_mask=self.slice_array(
                    "receptor/backbone_mask", rec_start, rec_end
                ),
                backbone=backbone,
                cif=system_info["rec_cif"],
            )

            pocket_backbone = BackboneData(
                coords=self.slice_array(
                    "pocket/backbone_coords", pocket_bb_start, pocket_bb_end
                ),
                res_ids=self.slice_array(
                    "pocket/backbone_res_ids", pocket_bb_start, pocket_bb_end
                ),
                res_names=self.slice_array(
                    "pocket/backbone_res_names", pocket_bb_start, pocket_bb_end
                ),
                chain_ids=self.slice_array(
                    "pocket/backbone_chain_ids", pocket_bb_start, pocket_bb_end
                ),
            )

            pocket = StructureData(
                coords=self.slice_array("pocket/coords", pocket_start, pocket_end),
                atom_names=self.slice_array(
                    "pocket/atom_names", pocket_start, pocket_end
                ),
                elements=self.slice_array("pocket/elements", pocket_start, pocket_end),
                res_ids=self.slice_array("pocket/res_ids", pocket_start, pocket_end),
                res_names=self.slice_array(
                    "pocket/res_names", pocket_start, pocket_end
                ),
                chain_ids=self.slice_array(
                    "pocket/chain_ids", pocket_start, pocket_end
                ),
                backbone_mask=self.slice_array(
                    "pocket/backbone_mask", pocket_start, pocket_end
                ),
                backbone=pocket_backbone,
            )
            npndes = None
            if system_info["npnde_idxs"]:
                npndes = self.get_npndes(system_info["npnde_idxs"])

            apo = None
            pred = None
            if link_type == "apo":
                apo_backbone = BackboneData(
                    coords=self.slice_array(
                        "apo/backbone_coords", link_bb_start, link_bb_end
                    ),
                    res_ids=None,
                    res_names=None,
                    chain_ids=None,
                )
                apo = StructureData(
                    coords=self.slice_array("apo/coords", link_start, link_end),
                    atom_names=None,
                    elements=None,
                    res_ids=None,
                    res_names=None,
                    chain_ids=None,
                    cif=system_info["link_cif"],
                    backbone_mask=None,
                    backbone=apo_backbone,
                )
            elif link_type == "pred":
                pred_backbone = BackboneData(
                    coords=self.slice_array(
                        "pred/backbone_coords", link_bb_start, link_bb_end
                    ),
                    res_ids=None,
                    res_names=None,
                    chain_ids=None,
                )
                pred = StructureData(
                    coords=self.slice_array("pred/coords", link_start, link_end),
                    atom_names=None,
                    elements=None,
                    res_ids=None,
                    res_names=None,
                    chain_ids=None,
                    cif=system_info["link_cif"],
                    backbone_mask=None,
                    backbone=pred_backbone,
                )
        else:
            apo = None
            pred = None

        is_covalent = False
        if system_info["linkages"]:
            is_covalent = True


        ligand = LigandData(
            sdf=system_info["lig_sdf"],
            ccd=system_info["ccd"],
            is_covalent=is_covalent,
            linkages=system_info["linkages"],
            coords=self.slice_array(
                "ligand/coords", lig_atom_start, lig_atom_end
            ),  # x
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

        if condensed_atom_typing:
            # Get extra ligand atom features
            lig_extra_feats = self.slice_array(f'ligand/extra_feats', lig_atom_start, lig_atom_end)
            lig_extra_feats = lig_extra_feats[:, :-1] # dangerous, implicit knowledge assumed about order/contents of extra_feats

            cond_a_typer = self.get_condensed_atom_typer()

            ligand.atom_cond_a = torch.from_numpy(cond_a_typer.feats_to_cond_a(ligand.atom_types, ligand.atom_charges, lig_extra_feats)).long() 

        elif include_extra_feats:
            # Get extra ligand atom features as a dictionary
            lig_extra_feats = self.slice_array(f'ligand/extra_feats', lig_atom_start, lig_atom_end)
            lig_extra_feats = lig_extra_feats[:, :-1]
            features = self.root['ligand/extra_feats'].attrs.get('features', [])

            lig_extra_feats_dict = {}

            # Iterate over all but the last feature
            for col_idx, feat in enumerate(features[:-1]):
                col_data = lig_extra_feats[:, col_idx]         
                lig_extra_feats_dict[feat] = torch.from_numpy(col_data).long()

            # add extra features to ligand
            ligand.atom_impl_H=lig_extra_feats_dict['impl_H']
            ligand.atom_aro=lig_extra_feats_dict['aro']
            ligand.atom_hyb=lig_extra_feats_dict['hyb']
            ligand.atom_ring=lig_extra_feats_dict['ring']
            ligand.atom_chiral=lig_extra_feats_dict['chiral']

        if include_pharmacophore:
            pharm_start, pharm_end = (
                system_info["pharm_start"],
                system_info["pharm_end"],
            )
            pharm_idxs = np.arange(pharm_start, pharm_end)
            interacting_pharms = pharm_idxs[self.slice_array("pharmacophore/interactions", pharm_start, pharm_end)]

            # TODO: what should we do if there are no interacting pharmacophores?
            if len(interacting_pharms) == 0:
                print(f"Warning: No interacting pharmacophores in system {index}.")
                pharmacophore = None
                
            else:
                pharm_sample_size =  np.random.randint(1, min(self.max_pharms_sampled, len(interacting_pharms)) + 1)
                pharm_sample = np.random.choice(interacting_pharms, size=pharm_sample_size, replace=False)

                coords = np.array([self.slice_array("pharmacophore/coords", i, i+1) for i in pharm_sample]).squeeze(1)
                types = np.array([self.slice_array("pharmacophore/types", i, i+1) for i in pharm_sample]).squeeze(1)
                vectors = np.array([self.slice_array("pharmacophore/vectors", i, i+1) for i in pharm_sample]).squeeze(1)
                interactions = np.ones(len(pharm_sample), dtype=bool)
                
                pharmacophore = PharmacophoreData(
                    coords=coords,
                    types=types,
                    vectors=vectors,
                    interactions=interactions
                )


        system = SystemData(
            system_id=system_info["system_id"],
            ligand_id=system_info["ligand_id"],
            receptor=receptor if include_protein else None,
            ligand=ligand,
            pharmacophore=pharmacophore if include_pharmacophore else None,
            pocket=pocket if include_protein else None,
            npndes=npndes if include_protein else None,
            link_type=link_type,
            link_id=system_info["link_id"] if link_type else None,
            link=apo if apo else pred,
        )
        return system

    def encode_atom_names(
        self,
        atom_names: np.ndarray,
        elements: np.ndarray,      # (unused here)
        res_names: np.ndarray      # (unused here)
    ) -> np.ndarray:
        # 1) find all the unique atom_names and the mapping back
        unique_names, inverse = np.unique(atom_names, return_inverse=True)

        # 2) do one dict-lookup per unique name
        unk_code = self.encode_atom["UNK"]
        unique_codes = np.array([
            self.encode_atom.get(name, unk_code)
            for name in unique_names
        ], dtype=np.int64)

        # 3) expand back out to the original shape
        return unique_codes[inverse]

    def encode_elements(self, elements: np.ndarray) -> np.ndarray:
        # Vectorized mapping of element symbols to integer codes
        unique_elems, inverse = np.unique(elements, return_inverse=True)
        unique_codes = np.array(
            [self.encode_element[elem] for elem in unique_elems],
            dtype=np.int64
        )
        return unique_codes[inverse]

    def encode_res_names(self, res_names: np.ndarray) -> np.ndarray:
        # Vectorized mapping of residue names to integer codes
        # 1. Extract uniques and inverse indices
        unique_names, inverse = np.unique(res_names, return_inverse=True)
        # 2. Map each unique name to its code (with substitutions for unknowns)
        unk_code = self.encode_residue["UNK"]
        unique_codes = np.array([
            self.encode_residue[name]
            if name in self.encode_residue
            else self.encode_residue.get(aa_substitutions.get(name, "UNK"), unk_code)
            for name in unique_names
        ], dtype=np.int64)
        # 3. Reconstruct full array via inverse mapping
        return unique_codes[inverse]

    def get_link_coords(
        self,
        link: StructureData,
        pocket_mask: torch.Tensor,
        bb_pocket_mask: torch.Tensor,
        modality_name: str,
    ) -> torch.Tensor:
        if modality_name == "prot_atom_x":
            x_0 = torch.from_numpy(link.coords[pocket_mask]).float()
        elif modality_name == "prot_res":
            x_0 = torch.from_numpy(link.backbone.coords[bb_pocket_mask]).float()
        else:
            raise NotImplementedError(
                f"{modality_name} does not have linked structure coords"
            )
        return x_0

    def convert_protein(
        self,
        holo: StructureData,
        pocket: StructureData,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        prot_coords = torch.from_numpy(holo.coords).float()
        prot_atom_names = torch.from_numpy(
            self.encode_atom_names(holo.atom_names, holo.elements, holo.res_names)
        ).long()
        prot_elements = torch.from_numpy(self.encode_elements(holo.elements)).long()
        prot_res_ids = torch.from_numpy(holo.res_ids).long()
        prot_res_names = torch.from_numpy(self.encode_res_names(holo.res_names)).long()
        prot_backbone_mask = torch.from_numpy(holo.backbone_mask).bool()

        # TODO: figure out how to store chain ids

        unique_chains = sorted(set(holo.chain_ids))
        chain_to_idx = {chain: idx for idx, chain in enumerate(unique_chains)}
        prot_chain_ids = torch.tensor(
            [chain_to_idx[chain_id] for chain_id in holo.chain_ids], dtype=torch.long
        )

        pocket_res_identifiers = set()
        for i in range(len(pocket.res_ids)):
            chain_id = pocket.chain_ids[i]
            res_id = pocket.res_ids[i]
            pocket_res_identifiers.add((chain_id, res_id))

        # 1. turn your pocket set into a small (M×2) int tensor
        pocket_pairs = [(chain_to_idx[c], r) for c, r in pocket_res_identifiers]
        pocket_pairs_tensor = torch.tensor(pocket_pairs, device=prot_res_ids.device, dtype=torch.long)  # (M,2)

        # 2. stack your per‐node (chain, res) into an (N×2) tensor
        prot_pairs = torch.stack((prot_chain_ids, prot_res_ids), dim=1)  # (N,2)

        # 3. compare all pairs at once, then reduce
        #    -> (N,1,2) == (1,M,2)  →  (N,M,2) boolean
        eq = prot_pairs.unsqueeze(1) == pocket_pairs_tensor.unsqueeze(0)
        #    want rows where *both* entries match, then any match across M
        pocket_mask = eq.all(dim=2).any(dim=1)   # (N,)

        node_data["prot_atom"] = {
            "x_1_true": prot_coords[pocket_mask],
            "a_1_true": prot_atom_names[pocket_mask],
            "e_1_true": prot_elements[pocket_mask],
            "res_id": prot_res_ids[pocket_mask],
            "res_names": prot_res_names[pocket_mask],
            "res_names_1_true": prot_res_names[pocket_mask],
            "chain_id": prot_chain_ids[pocket_mask],
            "backbone_mask": prot_backbone_mask[pocket_mask],
        }

        backbone_coords = torch.from_numpy(holo.backbone.coords).float()
        backbone_res_ids = torch.from_numpy(holo.backbone.res_ids).long()
        backbone_res_names = torch.from_numpy(
            self.encode_res_names(holo.backbone.res_names)
        ).long()

        # TODO: figure out how to store chain ids
        backbone_chain_ids = torch.tensor(
            [chain_to_idx[chain_id] for chain_id in holo.backbone.chain_ids],
            dtype=torch.long,
        )

        # Vectorized construction of backbone_pocket_mask
        # 1. Stack backbone (chain_idx, res_id) pairs into a (N_bb, 2) tensor
        backbone_pairs = torch.stack((backbone_chain_ids, backbone_res_ids), dim=1)
        # 2. Compare each backbone pair against all pocket pairs (broadcasted)
        eq_bb = backbone_pairs.unsqueeze(1) == pocket_pairs_tensor.unsqueeze(0)
        # 3. Mask true where both chain and residue match any pocket pair
        backbone_pocket_mask = eq_bb.all(dim=2).any(dim=1)

        node_data["prot_res"] = {
            "x_1_true": backbone_coords[backbone_pocket_mask],
            "res_id": backbone_res_ids[backbone_pocket_mask],
            "a_1_true": backbone_res_names[backbone_pocket_mask],
            "chain_id": backbone_chain_ids[backbone_pocket_mask],
        }

        return node_data, edge_idxs, edge_data, pocket_mask, backbone_pocket_mask

    def encode_charges(self, charges: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.charge_map_tensor, charges)
    
    def infer_covalent_bonds(
        self,
        ligand: LigandData,
        pocket: StructureData,
        atom_type_map: List[str],
        )   -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) pull out arrays
        lig = ligand.to_atom_array(atom_type_map)
        rec = pocket.to_atom_array()

        # element labels
        lig_elems = np.array([a.element for a in lig])        # shape (L,)
        rec_elems = np.array([a.element for a in rec])        # shape (R,)

        # 2) compute full distance matrix once
        dists = cdist(
            lig.coord,    # (L,1,3)
            rec.coord     # (1,R,3)
        )                              # → (L,R)

        # 3) build a mask of “allowed bonds” by iterating only DEFAULT_DISTANCE_RANGE
        masks = []
        for (e1, e2), (d_min, d_max) in DEFAULT_DISTANCE_RANGE.items():
            # ligand‐side eq e1, rec‐side eq e2
            m_l1 = (lig_elems == e1)    # (L,)
            m_r2 = (rec_elems == e2)    # (R,)
            # ligand‐side eq e2, rec‐side eq e1  (the “reverse” case)
            m_l2 = (lig_elems == e2)
            m_r1 = (rec_elems == e1)

            # skip if neither direction can possibly match
            if not ((m_l1.any() and m_r2.any()) or (m_l2.any() and m_r1.any())):
                continue

            # ligand e1 ↔ rec e2
            mask12 = m_l1[:, None] & m_r2[None, :]
            # ligand e2 ↔ rec e1
            mask21 = m_l2[:, None] & m_r1[None, :]

            # apply the same distance thresholds to both
            mask12 = mask12 & (dists >= d_min) & (dists <= d_max)
            mask21 = mask21 & (dists >= d_min) & (dists <= d_max)

            masks.append(mask12 | mask21)

        if not masks:
            # no possible bonds
            empty = torch.zeros((2,0), dtype=torch.long)
            return empty, empty

        # 4) combine all masks into one
        mask_all = np.logical_or.reduce(masks)  # shape (L,R)

        # 5) extract the i,j indices of True entries
        lig_idx, rec_idx = np.nonzero(mask_all)  # arrays of equal length N

        # 6) build atom-level edges
        prot_atom = torch.tensor(
            np.vstack([rec_idx, lig_idx]), dtype=torch.long
        )

        backbone = pocket.backbone
        res_index = {
            (rid, cid): idx
            for idx, (rid, cid) in enumerate(zip(backbone.res_ids, backbone.chain_ids))
        }

        # 7) map each rec_idx → residue index via our dict
        rec_res_ids   = rec.get_annotation("res_id")[rec_idx]
        rec_chain_ids = rec.get_annotation("chain_id")[rec_idx]
        res_idx = [res_index[(rid, cid)] for rid, cid in zip(rec_res_ids, rec_chain_ids) if (rid, cid) in res_index]
        
        # TODO: actually fix this bug 
        empty = torch.zeros((2,0), dtype=torch.long)
        return prot_atom, empty
        prot_res = torch.tensor(
            np.vstack([res_idx, lig_idx]), dtype=torch.long
        )
        
        return prot_atom, prot_res
        
    def convert_ligand(
        self,
        ligand: LigandData,
        ligand_id: str,
        task: Task,
        include_extra_feats: bool,
        condensed_atom_typing: bool,
        pocket: Optional[StructureData] = None,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:

        lig_xace = ligand.to_xace_mol(dense=True)

        denovo_ligand = any(group in task.groups_generated for group in ['ligand_identity',  'ligand_identity_condensed'])

        if self.fake_atom_p > 0 and denovo_ligand:
            if condensed_atom_typing:
                cond_a_typer = self.get_condensed_atom_typer()
                lig_xace = add_fake_atoms(lig_xace, self.fake_atom_p, cond_a_typer)
            else:
                lig_xace = add_fake_atoms(lig_xace, fake_atom_p=self.fake_atom_p)
        
        node_data = {
            "lig": {
                "x_1_true": lig_xace.x,
            }
        }

        if condensed_atom_typing:
            node_data["lig"]["cond_a_1_true"] = lig_xace.cond_a
        
        else:
            lig_c = self.encode_charges(lig_xace.c)
            node_data["lig"]["a_1_true"] = lig_xace.a
            node_data["lig"]["c_1_true"] = lig_c

            if include_extra_feats:
                node_data["lig"]["impl_H_1_true"] = lig_xace.impl_H
                node_data["lig"]["aro_1_true"] = lig_xace.aro
                node_data["lig"]["hyb_1_true"] = lig_xace.hyb
                node_data["lig"]["ring_1_true"] = lig_xace.ring
                node_data["lig"]["chiral_1_true"] = lig_xace.chiral

        edge_data = {
            "lig_to_lig": {
                "e_1_true": lig_xace.e,
            }
        }

        edge_idxs = {
            "lig_to_lig": lig_xace.edge_idxs,
        }
        if ligand.is_covalent and ligand.linkages and pocket is not None:
            prot_atom_to_lig_tensor, prot_res_to_lig_tensor = self.infer_covalent_bonds(
                ligand, pocket, lig_atom_type_map
            )
            if prot_atom_to_lig_tensor.shape[1] > 0:
                edge_idxs["prot_atom_covalent_lig"] = prot_atom_to_lig_tensor
                lig_to_prot_atom_tensor = prot_atom_to_lig_tensor[[1, 0]]
                edge_idxs["lig_covalent_prot_atom"] = lig_to_prot_atom_tensor

            if prot_res_to_lig_tensor.shape[1] > 0:
                edge_idxs["prot_res_covalent_lig"] = prot_res_to_lig_tensor
                lig_to_prot_res_tensor = prot_res_to_lig_tensor[[1, 0]]
                edge_idxs["lig_covalent_prot_res"] = lig_to_prot_res_tensor

        return node_data, edge_idxs, edge_data

    def convert_npndes(
        self,
        npndes: Dict[str, LigandData],
        pocket: Optional[StructureData] = None,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data, edge_data, edge_idxs = {}, {}, {}
        node_data["npnde"] = {
            "x_1_true": torch.empty(0),
            "a_1_true": torch.empty(0),
            "c_1_true": torch.empty(0),
        }
        edge_data["npnde_to_npnde"] = {"e_1_true": torch.empty(0)}
        edge_idxs["npnde_to_npnde"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["prot_atom_covalent_npnde"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["npnde_covalent_prot_atom"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["prot_res_covalent_npnde"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["npnde_covalent_prot_res"] = torch.empty((2, 0), dtype=torch.long)

        if not npndes:
            return node_data, edge_idxs, edge_data

        all_coords = []
        all_atom_types = []
        all_atom_charges = []
        all_bond_types = []
        all_bond_indices = []

        all_prot_atom_to_npnde_idxs = []
        all_prot_res_to_npnde_idxs = []

        node_offset = 0

        for npnde_id, ligand_data in npndes.items():
            coords = torch.from_numpy(ligand_data.coords).float()
            atom_types = torch.from_numpy(ligand_data.atom_types).long()
            atom_charges = torch.from_numpy(ligand_data.atom_charges).long()

            all_coords.append(coords)
            all_atom_types.append(atom_types)
            all_atom_charges.append(atom_charges)

            # check if the npnde has bonds
            has_bonds = (
                ligand_data.bond_types is not None
                and ligand_data.bond_indices is not None
            )
            if has_bonds and ligand_data.bond_types.shape[0] == 0:
                has_bonds = False

            if has_bonds:
                bond_types = torch.from_numpy(ligand_data.bond_types).long()
                bond_indices = torch.from_numpy(ligand_data.bond_indices).long()

                adjusted_indices = bond_indices.clone()
                adjusted_indices[:, 0] += node_offset
                adjusted_indices[:, 1] += node_offset

                all_bond_types.append(bond_types)
                all_bond_indices.append(adjusted_indices)

            if ligand_data.is_covalent and ligand_data.linkages and pocket is not None:
                prot_atom_to_npnde_tensor, prot_res_to_npnde_tensor = (
                    self.infer_covalent_bonds(ligand_data, pocket, npnde_atom_type_map)
                )
                if prot_atom_to_npnde_tensor.shape[1] > 0:
                    prot_atom_to_npnde_tensor[1, :] += node_offset
                    all_prot_atom_to_npnde_idxs.append(prot_atom_to_npnde_tensor)
                if prot_res_to_npnde_tensor.shape[1] > 0:
                    prot_res_to_npnde_tensor[1, :] += node_offset
                    all_prot_res_to_npnde_idxs.append(prot_res_to_npnde_tensor)

            node_offset += coords.shape[0]

        combined_coords = (
            torch.cat(all_coords, dim=0)
            if all_coords
            else torch.zeros((0, 3), dtype=torch.float)
        )
        combined_atom_types = (
            torch.cat(all_atom_types, dim=0)
            if all_atom_types
            else torch.zeros((0,), dtype=torch.long)
        )
        combined_atom_charges = (
            torch.cat(all_atom_charges, dim=0)
            if all_atom_charges
            else torch.zeros((0,), dtype=torch.long)
        )

        if all_bond_types and all_bond_indices:
            combined_bond_types = torch.cat(all_bond_types, dim=0)
            combined_bond_indices = torch.cat(all_bond_indices, dim=0)

            k = self.graph_config["edges"]["npnde_to_npnde"]["params"]["k"]
            npnde_x, npnde_a, npnde_c, npnde_e, npnde_edge_idxs = add_k_hop_edges(
                combined_coords,
                combined_atom_types,
                combined_atom_charges,
                combined_bond_types,
                combined_bond_indices,
                k=k,
            )
            npnde_c = self.encode_charges(npnde_c)

            node_data["npnde"] = {
                "x_1_true": npnde_x,
                "a_1_true": npnde_a,
                "c_1_true": npnde_c,
            }

            edge_data["npnde_to_npnde"] = {"e_1_true": npnde_e}

            edge_idxs["npnde_to_npnde"] = npnde_edge_idxs
        else:
            combined_atom_charges = self.encode_charges(combined_atom_charges)
            node_data["npnde"] = {
                "x_1_true": combined_coords,
                "a_1_true": combined_atom_types,
                "c_1_true": combined_atom_charges,
            }

        if all_prot_atom_to_npnde_idxs:
            prot_atom_to_npnde_tensor = torch.cat(all_prot_atom_to_npnde_idxs, dim=1)
            edge_idxs["prot_atom_covalent_npnde"] = prot_atom_to_npnde_tensor
            npnde_to_prot_atom_tensor = prot_atom_to_npnde_tensor[[1, 0]]
            edge_idxs["npnde_covalent_prot_atom"] = npnde_to_prot_atom_tensor

        if all_prot_res_to_npnde_idxs:
            prot_res_to_npnde_tensor = torch.cat(all_prot_res_to_npnde_idxs, dim=1)
            edge_idxs["prot_res_covalent_npnde"] = prot_res_to_npnde_tensor
            npnde_to_prot_res_tensor = prot_res_to_npnde_tensor[[1, 0]]
            edge_idxs["npnde_covalent_prot_res"] = npnde_to_prot_res_tensor

        return node_data, edge_idxs, edge_data

    def convert_pharmacophore(
        self, pharmacophore: PharmacophoreData
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        coords = torch.from_numpy(pharmacophore.coords).float()
        types = torch.from_numpy(pharmacophore.types).long()
        vectors = torch.from_numpy(pharmacophore.vectors).float()
        interactions = torch.from_numpy(pharmacophore.interactions).bool()

        node_data["pharm"] = {
            "x_1_true": coords,
            "a_1_true": types,
            "v_1_true": vectors,
            "i_1_true": interactions,
        }

        return node_data, edge_idxs, edge_data

    def convert_system(
        self, 
        system: SystemData,
        task: Task, 
        include_pharmacophore: bool, 
        include_protein: bool,
        include_extra_feats: bool,
        condensed_atom_typing: bool
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        pocket_mask = None
        bb_pocket_mask = None

        # read protein data
        if include_protein:
            (
                prot_node_data,
                prot_edge_idxs,
                prot_edge_data,
                pocket_mask,
                bb_pocket_mask,
            ) = self.convert_protein(system.receptor, system.pocket)
            node_data.update(prot_node_data)
            edge_idxs.update(prot_edge_idxs)
            edge_data.update(prot_edge_data)

            # read npnde data
            npnde_node_data, npnde_edge_idxs, npnde_edge_data = self.convert_npndes(
                system.npndes if system.npndes is not None else {}, system.pocket
            )
            node_data.update(npnde_node_data)
            edge_idxs.update(npnde_edge_idxs)
            edge_data.update(npnde_edge_data)

        # read ligand data
        lig_node_data, lig_edge_idxs, lig_edge_data = self.convert_ligand(
            system.ligand, 
            system.ligand_id, 
            task=task,
            include_extra_feats=include_extra_feats,
            condensed_atom_typing=condensed_atom_typing,
            pocket=system.pocket
        )
        node_data.update(lig_node_data)
        edge_idxs.update(lig_edge_idxs)
        edge_data.update(lig_edge_data)

        if include_pharmacophore and system.pharmacophore is not None:
            pharm_node_data, pharm_edge_idxs, pharm_edge_data = (
                self.convert_pharmacophore(system.pharmacophore)
            )
            node_data.update(pharm_node_data)
            edge_idxs.update(pharm_edge_idxs)
            edge_data.update(pharm_edge_data)

        return node_data, edge_idxs, edge_data, pocket_mask, bb_pocket_mask

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class(task_name)

        include_pharmacophore = "pharmacophore" in task_class.groups_present
        include_extra_feats = "ligand_identity_extra" in task_class.groups_present
        condensed_atom_typing = "ligand_identity_condensed" in task_class.groups_present

        include_protein = (
            "protein_identity" in task_class.groups_present
            or "protein_structure" in task_class.groups_present
        )

        system = self.get_system(
            idx,
            include_pharmacophore=include_pharmacophore,
            include_protein=include_protein,
            include_extra_feats=include_extra_feats,
            condensed_atom_typing=condensed_atom_typing,
        )

        node_data, edge_idxs, edge_data, pocket_mask, bb_pocket_mask = (
            self.convert_system(
                system,
                task=task_class,
                include_pharmacophore=include_pharmacophore,
                include_protein=include_protein,
                include_extra_feats=include_extra_feats,
                condensed_atom_typing=condensed_atom_typing
            )
        )

        g = build_complex_graph(
            node_data,
            edge_idxs,
            edge_data,
            task=task_class,
            graph_config=self.graph_config,
        )

        if include_protein:
            # standardize residue ids (ensure all residue ids start at 0 across chains)
            # node_data contains chain_ds, and prot_res_ids
            prot_res_ids = node_data["prot_atom"]["res_id"] #starting indices 
            prot_chain_ids = node_data["prot_atom"]["chain_id"] # protein chain ids
            residue_idxs = self.standardize_residue_ids(prot_res_ids, prot_chain_ids)
            # create protein position embeddings
            protein_position_encodings = residue_sinusoidal_encoding(residue_idxs, self.res_id_embed_dim)
            
            # Add the position embeddings to the graph's protein atom nodes
            g.nodes["prot_atom"].data["pos_enc_1_true"] = protein_position_encodings

        # center ground truth coordinates on ligand
        g = center_on_ligand_gt(g)
        # apply random rotation to ground truth coordinates
        g = rotate_ground_truth(g)

        # apply system offset
        # TODO: expose as a config parameter
        if self.sys_offset_std > 0:
            g = system_offset(g, offset_std=self.sys_offset_std)

        # get prior functions
        prior_fns = get_prior(task_class, self.prior_config, training=True)

        # first, if the task requires a linked structure for the prior,
        # manually add this to the graph

        if "apo" in prior_fns.get("prot_atom_x", ("", None))[0]:
            if system.link is None:
                raise ValueError(
                    "system.link is None, cannot retrieve link coordinates."
                )

            g.nodes["prot_atom"].data["x_0"] = self.get_link_coords(
                system.link, pocket_mask, bb_pocket_mask, "prot_atom_x"
            )

        # sample priors
        g = sample_priors(g, 
                          task_class=task_class, 
                          prior_fns=prior_fns, 
                          training=True, 
                          fake_atoms=self.use_fake_atoms)

        return g

    def retrieve_graph_chunks(self, frac_start, frac_end, apo_systems: bool = False):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk.
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self)  # this is wrong! n_graphs depends on apo_systems!!!!
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        # raise NotImplementedError(
        #     "need to build capability to modify chunks based on whether or not the task uses the apo state"
        # )

        # construct a tensor containing the index ranges for each chunk
        chunk_index = torch.zeros(n_chunks, 2, dtype=torch.int64)
        chunk_index[:, 0] = self.graphs_per_chunk * torch.arange(n_chunks)
        chunk_index[:-1, 1] = chunk_index[1:, 0]
        chunk_index[-1, 1] = n_graphs

        return chunk_index

    def get_num_nodes(self, task: Task, start_idx, end_idx, per_ntype=False):
        # here, unlike in other places, start_idx and end_idx are
        # indexes into the system_lookup array, not a node/edge data array

        node_types = ["lig", "prot_atom", "prot_res", "npnde"]
        if "pharmacophore" in task.groups_present:
            node_types.append("pharm")

        node_counts = []
        for ntype in node_types:
            if ntype == "lig":
                counts = np.array(
                    [
                        row["lig_atom_end"] - row["lig_atom_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "prot_atom":
                counts = np.array(
                    [
                        row["pocket_end"] - row["pocket_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "prot_res":
                counts = np.array(
                    [
                        row["pocket_bb_end"] - row["pocket_bb_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "npnde":
                counts = []
                for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                    "records"
                ):
                    npnde_count = 0
                    if row["npnde_idxs"]:
                        for npnde_idx in row["npnde_idxs"]:
                            npnde_row = self.npnde_lookup.iloc[npnde_idx]
                            npnde_count += (
                                npnde_row["atom_end"] - npnde_row["atom_start"]
                            )
                    counts.append(npnde_count)
                counts = np.array(counts)
            elif ntype == "pharm":
                counts = []
                
                for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                    "records"
                ):
                    n_gt_pharms = row["pharm_end"] - row["pharm_start"]
                    expected_n_pharms = int(round(min(self.max_pharms_sampled, n_gt_pharms) / 2))   # we sample between 1 and max_pharms_sampled pharmacophores, so (1+max_pharms_sampled)/2 pharms in expectation
                    counts.append(expected_n_pharms)

                counts = np.array(counts)

            node_counts.append(counts)

        if self.fake_atom_p > 0:
            lig_node_counts = node_counts[0]
            mean_fake_atoms = self.fake_atom_p/2 * lig_node_counts
            node_counts[0] = lig_node_counts + mean_fake_atoms.astype(int)

        if per_ntype:
            num_nodes_dict = {
                ntype: ncount for ntype, ncount in zip(node_types, node_counts)
            }
            return num_nodes_dict

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts

    def get_pskip(self, start_idx, end_idx):
        """
        Computes the p_skip values for the systems in the specified range.
        """
        pskip = self.system_lookup['p_skip'].values[start_idx:end_idx]
        pskip = torch.from_numpy(pskip)
        return pskip

    def retrieve_atom_idxs(self, index: int) -> Tuple:
        """ 
        Returns the starting and ending atom indices for the ligand 
        """
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == index
        ].iloc[0]

        lig_atom_start, lig_atom_end = (
            int(system_info["lig_atom_start"]),
            int(system_info["lig_atom_end"]),
        )
        return lig_atom_start, lig_atom_end
    
    def standardize_residue_ids(self, res_ids: torch.Tensor, chain_ids: torch.Tensor):
        """
        Normalize residue IDs within each chain by subtracting the chain-specific minimum residue ID
        """
        standardized = torch.zeros_like(res_ids)
        unique_chains = torch.unique(chain_ids, sorted=True)

        for chain in unique_chains:
            chain_mask = chain_ids == chain
            chain_res_ids = res_ids[chain_mask]
            
            min_res_id = torch.min(chain_res_ids)
            standardized[chain_mask] = chain_res_ids - min_res_id

        return standardized

def compute_pskip(
    df: pd.DataFrame,
    id_col: str = "cluster_id",
    freq_col: str = "freq",
    pskip_factor: float = 0.0,
) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame containing the frequency of each CCD code.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame with a column named `ccd_col`.
    ccd_col : str, default "CCD"
        Name of the column containing CCD codes.
    freq_col : str, default "CCD_freq"
        Name of the new column to create for frequencies.
    include_nan : bool, default False
        Whether to count NaN values in the frequency. If False, NaNs get a 0.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an extra column `freq_col`.
    """
    # Compute counts for each CCD value (optionally including NaNs)
    counts = df[id_col].value_counts(normalize=True)

    # Map counts back to the original rows
    df[freq_col] = df[id_col].map(counts)

    df['p_skip'] = 1 - df[freq_col].min() / df[freq_col]
    df['p_skip'] = df['p_skip'] * pskip_factor
    return df