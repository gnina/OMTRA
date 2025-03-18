import zarr
import logging
import argparse
from numcodecs import VLenUTF8
from omtra_pipelines.plinder_dataset.plinder_pipeline import *
from omtra_pipelines.plinder_dataset.plinder_links_zarr import *
from omtra_pipelines.plinder_dataset.zarr_retriever import PlinderZarrRetriever
from omtra_pipelines.plinder_dataset.utils import LIGAND_MAP, NPNDE_MAP
from plinder.core import PlinderSystem


def check_receptor(system: SystemData, actual_system: SystemData):
    assert system.receptor.cif == actual_system.receptor.cif, (
        f"Receptor cif mismatch zarr: {system.receptor.cif} processor {actual_system.receptor.cif}: {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.receptor.coords == actual_system.receptor.coords), (
        f"Receptor coords mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.receptor.atom_names == actual_system.receptor.atom_names), (
        f"Receptor atom_names mismatch {system.system_id}"
    )
    assert np.all(system.receptor.elements == actual_system.receptor.elements), (
        f"Receptor elements mismatch {system.system_id}"
    )
    assert np.all(system.receptor.res_ids == actual_system.receptor.res_ids), (
        f"Receptor res_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.receptor.res_names == system.receptor.res_names), (
        f"Receptor res_names mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.receptor.chain_ids == actual_system.receptor.chain_ids), (
        f"Receptor chain_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    assert np.all(
        system.receptor.backbone_mask == actual_system.receptor.backbone_mask
    ), (
        f"Receptor backbone_mask mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    assert np.all(
        system.receptor.backbone.coords == actual_system.receptor.backbone.coords
    ), (
        f"Receptor backbone coords mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.receptor.backbone.res_ids == actual_system.receptor.backbone.res_ids
    ), (
        f"Receptor backbone res_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.receptor.backbone.res_names == system.receptor.backbone.res_names
    ), (
        f"Receptor backbone res_names mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.receptor.backbone.chain_ids == actual_system.receptor.backbone.chain_ids
    ), (
        f"Receptor backbone chain_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )


def check_pocket(system: SystemData, actual_system: SystemData):
    assert len(system.pocket.coords) == len(actual_system.pocket.coords), (
        f"Pocket size mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    system_atoms = {}
    actual_system_atoms = {}

    for i in range(len(system.pocket.coords)):
        atom_id = (
            system.pocket.res_ids[i],
            system.pocket.res_names[i],
            system.pocket.atom_names[i],
            system.pocket.elements[i],
            system.pocket.chain_ids[i],
            system.pocket.backbone_mask[i],
        )
        system_atoms[atom_id] = system.pocket.coords[i]

    for i in range(len(actual_system.pocket.coords)):
        atom_id = (
            actual_system.pocket.res_ids[i],
            actual_system.pocket.res_names[i],
            actual_system.pocket.atom_names[i],
            actual_system.pocket.elements[i],
            actual_system.pocket.chain_ids[i],
            actual_system.pocket.backbone_mask[i],
        )
        actual_system_atoms[atom_id] = actual_system.pocket.coords[i]

    system_atom_ids = set(system_atoms.keys())
    actual_system_atom_ids = set(actual_system_atoms.keys())

    assert system_atom_ids == actual_system_atom_ids, (
        f"Pocket atom composition mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    for atom_id in system_atom_ids:
        assert np.allclose(
            system_atoms[atom_id], actual_system_atoms[atom_id], atol=1e-6
        ), (
            f"Coordinate mismatch for atom {atom_id} in {system.system_id} {system.ligand_id} {system.link_id}"
        )


def check_ligand(system: SystemData, actual_system: SystemData):
    assert np.all(system.ligand.coords == actual_system.ligand.coords), (
        f"Ligand coords mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.ligand.atom_types == actual_system.ligand.atom_types), (
        f"Ligand atom types mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.ligand.atom_charges == actual_system.ligand.atom_charges), (
        f"Ligand atom charges mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.ligand.bond_types == actual_system.ligand.bond_types), (
        f"Ligand bond types mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.ligand.bond_indices == actual_system.ligand.bond_indices), (
        f"Ligand bond indices mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert system.ligand.sdf == actual_system.ligand.sdf, (
        f"Ligand sdf mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert system.ligand.ccd == actual_system.ligand.ccd, (
        f"Ligand ccd mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert system.ligand.linkages == actual_system.ligand.linkages, (
        f"Ligand linkages mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert system.ligand.is_covalent == actual_system.ligand.is_covalent, (
        f"Ligand is_covalent mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )


def check_pharmacophore(system: SystemData, actual_system: SystemData):
    assert len(system.pharmacophore.coords) == len(
        actual_system.pharmacophore.coords
    ), (
        f"Pharmacophore size mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    system_features = []
    actual_system_features = []

    for i in range(len(system.pharmacophore.coords)):
        coords_tuple = tuple(float(x) for x in system.pharmacophore.coords[i])
        vector_tuple = None
        if system.pharmacophore.vectors[i] is not None:
            vector_tuple = tuple(
                float(x) for x in system.pharmacophore.vectors[i].flatten()
            )

        feature = (
            coords_tuple,
            system.pharmacophore.types[i],
            vector_tuple,
            system.pharmacophore.interactions[i],
        )
        system_features.append(feature)

    for i in range(len(actual_system.pharmacophore.coords)):
        coords_tuple = tuple(float(x) for x in actual_system.pharmacophore.coords[i])
        vector_tuple = None
        if actual_system.pharmacophore.vectors[i] is not None:
            vector_tuple = tuple(
                float(x) for x in actual_system.pharmacophore.vectors[i].flatten()
            )

        feature = (
            coords_tuple,
            actual_system.pharmacophore.types[i],
            vector_tuple,
            actual_system.pharmacophore.interactions[i],
        )
        actual_system_features.append(feature)

    system_features.sort()
    actual_system_features.sort()

    for i in range(len(system_features)):
        sys_feat = system_features[i]
        actual_feat = actual_system_features[i]

        coords_match = sys_feat[0] == actual_feat[0] or np.allclose(
            sys_feat[0], actual_feat[0], atol=1e-6
        )
        types_match = sys_feat[1] == actual_feat[1]

        if sys_feat[2] is not None and actual_feat[2] is not None:
            vectors_match = sys_feat[2] == actual_feat[2] or np.allclose(
                sys_feat[2], actual_feat[2], atol=1e-6
            )
        else:
            vectors_match = sys_feat[2] == actual_feat[2]

        interactions_match = sys_feat[3] == actual_feat[3]

        assert coords_match and types_match and vectors_match and interactions_match, (
            f"Pharmacophore feature mismatch at position {i} in {system.system_id} {system.ligand_id} {system.link_id}"
        )


def check_npndes(system: SystemData, actual_system: SystemData):
    for id, npnde in system.npndes.items():
        assert np.all(npnde.coords == actual_system.npndes[id].coords), (
            f"npnde {id} coords mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert np.all(npnde.atom_types == actual_system.npndes[id].atom_types), (
            f"npnde {id} atom types mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert np.all(npnde.atom_charges == actual_system.npndes[id].atom_charges), (
            f"npnde {id} atom charges mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert np.all(npnde.bond_types == actual_system.npndes[id].bond_types), (
            f"npnde {id} bond types mismatch {system.system_id} {system.ligand_id} {system.link_id} {npnde.bond_types} {actual_system.npndes[id].bond_types}"
        )
        assert np.all(npnde.bond_indices == actual_system.npndes[id].bond_indices), (
            f"npnde {id} bond indices mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert npnde.sdf == actual_system.npndes[id].sdf, (
            f"npnde {id} sdf mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert npnde.ccd == actual_system.npndes[id].ccd, (
            f"npnde {id} ccd mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert npnde.is_covalent == actual_system.npndes[id].is_covalent, (
            f"npnde {id} is covalent mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )
        assert npnde.linkages == actual_system.npndes[id].linkages, (
            f"npnde {id} linkages mismatch {system.system_id} {system.ligand_id} {system.link_id}"
        )


def check_link(system: SystemData, actual_system: SystemData):
    assert np.all(system.link.coords == actual_system.link.coords), (
        f"link coord mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.atom_names == actual_system.link.atom_names), (
        f"link atom names indices mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.elements == actual_system.link.elements), (
        f"link elements indices mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.res_ids == actual_system.link.res_ids), (
        f"link res ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.res_names == actual_system.link.res_names), (
        f"link res names mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.chain_ids == actual_system.link.chain_ids), (
        f"link chain ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.backbone_mask == actual_system.link.backbone_mask), (
        f"link backbone_mask mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert system.link.cif == actual_system.link.cif, (
        f"link cif mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    assert np.all(system.link.backbone.coords == actual_system.link.backbone.coords), (
        f"link backbone coords mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.link.backbone.res_ids == actual_system.link.backbone.res_ids
    ), (
        f"link backbone res_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.backbone.res_names == system.link.backbone.res_names), (
        f"link backbone res_names mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.link.backbone.chain_ids == actual_system.link.backbone.chain_ids
    ), (
        f"link backbone chain_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    # check correspondence with receptor
    assert np.all(system.link.atom_names == system.receptor.atom_names), (
        f"Atom names mismatch between link + receptor {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.elements == system.receptor.elements), (
        f"Elements mismatch between link + receptor {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.res_ids == system.receptor.res_ids), (
        f"res ids mismatch between link + receptor {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.res_names == system.receptor.res_names), (
        f"res names mismatch between link + receptor {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(system.link.chain_ids == system.receptor.chain_ids), (
        f"chain ids mismatch between link + receptor {system.system_id} {system.ligand_id} {system.link_id}"
    )

    assert np.all(system.link.backbone.res_ids == system.receptor.backbone.res_ids), (
        f"link + receptor backbone res_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.link.backbone.res_names == system.receptor.backbone.res_names
    ), (
        f"link+receptor backbone res_names mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )
    assert np.all(
        system.link.backbone.chain_ids == system.receptor.backbone.chain_ids
    ), (
        f"link+receptor backbone chain_ids mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )

    assert np.all(system.link.backbone_mask == system.receptor.backbone_mask), (
        f"link+receptor backbone_mask mismatch {system.system_id} {system.ligand_id} {system.link_id}"
    )


def check_system(system: SystemData, actual_system: SystemData):
    check_receptor(system, actual_system)
    check_ligand(system, actual_system)
    check_pocket(system, actual_system)
    check_pharmacophore(system, actual_system)
    if system.npndes:
        check_npndes(system, actual_system)
    if system.link_type:
        check_link(system, actual_system)


def check_storage(zarr_path):
    retriever = PlinderZarrRetriever(zarr_path=zarr_path)

    n = retriever.get_length()
    print(f"{n} systems to check")

    for i in range(n):
        print(f"Checking index {i}")
        system = retriever.get_system(i)
        ligand_id = system.ligand_id
        link_id = system.link_id
        if system.link_type == "apo":
            processor = SystemProcessor(system_id=system.system_id, link_type="apo")
            result = processor.process_system()
        elif system.link_type == "pred":
            processor = SystemProcessor(system_id=system.system_id, link_type="pred")
            result = processor.process_system()
        actual_system = None
        for system_data in result[system.link_type][link_id]:
            if system_data.ligand_id == ligand_id:
                actual_system = system_data
        assert actual_system is not None
        check_system(system, actual_system)
        print(f"Passed")


def main():
    parser = argparse.ArgumentParser(description="test plinder storage")
    parser.add_argument(
        "--zarr_path", type=str, required=True, help="path to zarr store"
    )
    args = parser.parse_args()

    check_storage(args.zarr_path)


if __name__ == "__main__":
    main()
