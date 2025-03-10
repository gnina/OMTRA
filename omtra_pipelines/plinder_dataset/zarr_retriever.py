import logging
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import zarr
from numcodecs import VLenUTF8
from omtra_pipelines.plinder_dataset.plinder_pipeline import (
    LigandData,
    StructureData,
    PharmacophoreData,
    SystemData,
    SystemProcessor,
)
from tqdm import tqdm


class PlinderZarrRetriever:
    def __init__(self, zarr_path: str):
        assert Path(zarr_path).exists()

        self.zarr_path = zarr_path
        self.root = zarr.open_group(store=self.zarr_path, mode="r+")

        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])
        self.npnde_lookup = pd.DataFrame(self.root.attrs["npnde_lookup"])

    def get_length(self) -> int:
        return self.system_lookup.shape[0]

    def get_npndes(self, npnde_idxs: List[int]) -> Dict[str, LigandData]:
        npndes = {}
        for idx in npnde_idxs:
            npnde_info = self.npnde_lookup[self.npnde_lookup["npnde_idx"] == idx].iloc[
                0
            ]

            key = npnde_info["npnde_id"]

            atom_start, atom_end = npnde_info["atom_start"], npnde_info["atom_end"]
            bond_start, bond_end = npnde_info["bond_start"], npnde_info["bond_end"]

            is_covalent = False
            if npnde_info["linkages"]:
                is_covalent = True

            npndes[key] = LigandData(
                sdf=npnde_info["lig_sdf"],
                ccd=npnde_info["ccd"],
                is_covalent=is_covalent,
                linkages=npnde_info["linkages"],
                coords=self.root["npnde"]["coords"][atom_start:atom_end],
                atom_types=self.root["npnde"]["atom_types"][atom_start:atom_end],
                atom_charges=self.root["npnde"]["atom_charges"][atom_start:atom_end],
                bond_types=self.root["npnde"]["bond_types"][bond_start:bond_end],
                bond_indices=self.root["npnde"]["bond_indices"][bond_start:bond_end],
            )
        return npndes

    def get_system(self, index: int) -> SystemData:
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == index
        ].iloc[0]

        rec_start, rec_end = system_info["rec_start"], system_info["rec_end"]
        lig_atom_start, lig_atom_end = (
            system_info["lig_atom_start"],
            system_info["lig_atom_end"],
        )
        lig_bond_start, lig_bond_end = (
            system_info["lig_bond_start"],
            system_info["lig_bond_end"],
        )
        pharm_start, pharm_end = system_info["pharm_start"], system_info["pharm_end"]
        pocket_start, pocket_end = (
            system_info["pocket_start"],
            system_info["pocket_end"],
        )
        link_start, link_end = system_info["link_start"], system_info["link_end"]
        link_type = system_info["link_type"]

        receptor = StructureData(
            coords=self.root["receptor"]["coords"][rec_start:rec_end],
            atom_names=self.root["receptor"]["atom_names"][rec_start:rec_end].astype(
                str
            ),
            elements=self.root["receptor"]["elements"][rec_start:rec_end].astype(str),
            res_ids=self.root["receptor"]["res_ids"][rec_start:rec_end],
            res_names=self.root["receptor"]["res_names"][rec_start:rec_end].astype(str),
            chain_ids=self.root["receptor"]["chain_ids"][rec_start:rec_end].astype(str),
            cif=system_info["rec_cif"],
        )

        is_covalent = False
        if system_info["linkages"]:
            is_covalent = True

        ligand = LigandData(
            sdf=system_info["lig_sdf"],
            ccd=system_info["ccd"],
            is_covalent=is_covalent,
            linkages=system_info["linkages"],
            coords=self.root["ligand"]["coords"][lig_atom_start:lig_atom_end],
            atom_types=self.root["ligand"]["atom_types"][lig_atom_start:lig_atom_end],
            atom_charges=self.root["ligand"]["atom_charges"][
                lig_atom_start:lig_atom_end
            ],
            bond_types=self.root["ligand"]["bond_types"][lig_bond_start:lig_bond_end],
            bond_indices=self.root["ligand"]["bond_indices"][
                lig_bond_start:lig_bond_end
            ],
        )

        pharmacophore = PharmacophoreData(
            coords=self.root["pharmacophore"]["coords"][pharm_start:pharm_end],
            types=self.root["pharmacophore"]["types"][pharm_start:pharm_end],
            vectors=self.root["pharmacophore"]["vectors"][pharm_start:pharm_end],
            interactions=self.root["pharmacophore"]["interactions"][
                pharm_start:pharm_end
            ],
        )

        pocket = StructureData(
            coords=self.root["pocket"]["coords"][pocket_start:pocket_end],
            atom_names=self.root["pocket"]["atom_names"][
                pocket_start:pocket_end
            ].astype(str),
            elements=self.root["pocket"]["elements"][pocket_start:pocket_end].astype(
                str
            ),
            res_ids=self.root["pocket"]["res_ids"][pocket_start:pocket_end],
            res_names=self.root["pocket"]["res_names"][pocket_start:pocket_end].astype(
                str
            ),
            chain_ids=self.root["pocket"]["chain_ids"][pocket_start:pocket_end].astype(
                str
            ),
        )
        npndes = None
        if system_info["npnde_idxs"]:
            npndes = self.get_npndes(system_info["npnde_idxs"])

        apo = None
        pred = None
        if link_type == "apo":
            apo = StructureData(
                coords=self.root["apo"]["coords"][link_start:link_end],
                atom_names=self.root["apo"]["atom_names"][link_start:link_end].astype(
                    str
                ),
                elements=self.root["apo"]["elements"][link_start:link_end].astype(str),
                res_ids=self.root["apo"]["res_ids"][link_start:link_end],
                res_names=self.root["apo"]["res_names"][link_start:link_end].astype(
                    str
                ),
                chain_ids=self.root["apo"]["chain_ids"][link_start:link_end].astype(
                    str
                ),
                cif=system_info["link_cif"],
            )
        elif link_type == "pred":
            pred = StructureData(
                coords=self.root["pred"]["coords"][link_start:link_end],
                atom_names=self.root["pred"]["atom_names"][link_start:link_end].astype(
                    str
                ),
                elements=self.root["pred"]["elements"][link_start:link_end].astype(str),
                res_ids=self.root["pred"]["res_ids"][link_start:link_end],
                res_names=self.root["pred"]["res_names"][link_start:link_end].astype(
                    str
                ),
                chain_ids=self.root["pred"]["chain_ids"][link_start:link_end].astype(
                    str
                ),
                cif=system_info["link_cif"],
            )

        system = SystemData(
            system_id=system_info["system_id"],
            ligand_id=system_info["ligand_id"],
            receptor=receptor,
            ligand=ligand,
            pharmacophore=pharmacophore,
            pocket=pocket,
            npndes=npndes,
            link_type=link_type,
            link_id=system_info["link_id"],
            link=apo if apo else pred,
        )
        return system
