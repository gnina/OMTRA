# this script creates a PyMOL command to visualize pharmacophores and their vectors.
# Usage:
# 1. Open PyMOL
# 2. Load this script via `run visualize_pharms.py`
# 3. Run `show_pharmvecs` with the path to an SDF file as the argument: `show_pharmvecs /path/to/file.sdf`
# requires pymol open source be installed inside omtra environment


from __future__ import print_function
# from pymol.cgo import *    # get constants
from pymol import cgo
from pymol import cmd
import numpy as np
from pathlib import Path

from omtra.data.pharmacophores import get_pharmacophores
from omtra.constants import ph_idx_to_type, ph_type_to_idx
from rdkit import Chem


type_idx_to_elem = ['P', 'S', 'F', 'N', 'O', 'C', 'Cl']
ph_type_to_elem = {ph_idx_to_type[i]: type_idx_to_elem[i] for i in range(len(ph_idx_to_type))}

def pharm_to_xyz(pos, types, ph_type_to_elem):
    out = f'{len(pos)}\n'
    for i in range(len(pos)):
        elem = ph_type_to_elem[types[i]]
        out += f"{elem} {pos[i, 0]:.3f} {pos[i, 1]:.3f} {pos[i, 2]:.3f}\n"
    return out

def pharm_to_file(pos, types, ph_type_to_elem, filename):

    types = [ ph_idx_to_type[i] for i in types]

    block = pharm_to_xyz(pos, types, ph_type_to_elem)
    with open(filename, 'w') as f:
        f.write(block)


def show_pharmvecs(
                sdf_path: str, 
                outname="modevectors", 
                headrgb="1.0,1.0,1.0", 
                tailrgb="1.0,1.0,1.0",  ):
    headrgb = headrgb.strip('" []()')
    tailrgb = tailrgb.strip('" []()')
    headrgb = list(map(float, headrgb.split(',')))
    tailrgb = list(map(float, tailrgb.split(',')))


    # compute pharmacophore, write to pharm.xyz
    mol_name = Path(sdf_path).stem
    mol = Chem.MolFromMolFile(sdf_path)
    x, a, vectors, _ = get_pharmacophores(mol)
    pharm_to_file(x, a, ph_type_to_elem, 'pharm.xyz')

    cmd.load(sdf_path) # load molecule
    cmd.load('pharm.xyz') # load pharmacophore
    cmd.set('sphere_scale', 0.4, 'pharm')
    cmd.set('sphere_transparency', 0.3, 'pharm')
    cmd.show('spheres', 'pharm')

    # create selection groups for each pharamcophore type
    for ph_type, elem in ph_type_to_elem.items():
        cmd.select(ph_type, f"pharm and elem {elem}")


    obj = cmd.get_model('pharm')

    # get coordinates of the object
    coords = []
    for atom in obj.atom:
        coords.append(atom.coord)
    coords = np.array(coords)

    assert coords.shape[0] == vectors.shape[0], f"Number of atoms ({coords.shape[0]}) in the object and the number of vectors must be the same ({vectors.shape[0]})  "

    arrow_cgos = []
    for coord_idx in range(coords.shape[0]):

        for vec_idx in range(vectors.shape[1]):
            v = vectors[coord_idx, vec_idx]
            if (v == 0).all():
                continue

            p1 = coords[coord_idx]
            p2 = p1 + v*2

            cylinder_end = (p2 - p1)*0.6 + p1
            cylinder_radius = 0.2
            arrow_head_radius = 0.0
            arrow_tail_radius = cylinder_radius*1.5

            tail = [
                # Tail of cylinder
                cgo.CYLINDER, *p1, *cylinder_end, cylinder_radius, *tailrgb, *tailrgb
            ]

            head = [
                cgo.CONE, *cylinder_end, *p2,
                arrow_tail_radius, arrow_head_radius, *headrgb, *headrgb, 1.0, 1.0]
            arrow_cgos = arrow_cgos + tail + head

    cmd.delete(outname)
    cmd.load_cgo(arrow_cgos, outname)  # Ray tracing an empty object will cause a segmentation fault.  No arrows = Do not display in PyMOL!!!
    
    # TODO: may need to manually display the object, not sure if load_cgo somehow affects the display
    # of other objects but this was just present in modevectors.py
    # cmd.show(representation="cartoon", selection=)





cmd.extend("show_pharmvecs", show_pharmvecs)