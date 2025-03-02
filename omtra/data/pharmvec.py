import numpy as np
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils

def GetAromaticFeatVects(atomsLoc, featLoc, return_both: bool = False):
    """Compute the direction vector for an aromatic feature."""
    
    v1 = atomsLoc[0] - featLoc
    v2 = atomsLoc[1] - featLoc

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    if return_both:
        normal2 = normal * -1
        return [normal, normal2]
    else:
        return [normal]


def GetDonorFeatVects(featAtoms, atomsLoc, rdmol):
    atom_idx = featAtoms[0]
    atom_coords = atomsLoc[0]
    vectors = []
    
    for nbor in rdmol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nbor.GetAtomicNum() == 1:  # hydrogen atom
            nbor_coords = np.array(rdmol.GetConformer().GetAtomPosition(nbor.GetIdx()))
            vec = nbor_coords - atom_coords
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        
    return vectors


def GetAcceptorFeatVects(featAtoms, atomsLoc, rdmol):
    atom_idx = featAtoms[0]
    atom_coords = atomsLoc[0]
    atom = rdmol.GetAtomWithIdx(atom_idx)
    nbrs = atom.GetNeighbors()
    conf = rdmol.GetConformer()
    
    ''' # pharmit
    vectors = []
    found_vec = False

    # check if any hydrogen neighbor exists
    for nbor in rdmol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if nbor.GetAtomicNum() == 1:
            nbor_coords = np.array(rdmol.GetConformer().GetAtomPosition(nbor.GetIdx()))
            vec = nbor_coords - atom_coords
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            found_vec = True

    '''
    
    # from rdkit/Chem/Features/FeatDirUtilsRD.py
    
    hydrogens = []
    heavy = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hydrogens.append(nbr)
        else:
            heavy.append(nbr)

    cpt = conf.GetAtomPosition(atom_idx)
    
    if atom.GetAtomicNum() == 8 and len(nbrs) < 3: # two lone pairs
        heavy_nbr = heavy[0]
        if len(nbrs) == 1: # sp2
            for a in heavy_nbr.GetNeighbors():
                if a.GetIdx() != atom_idx:
                    heavy_nbr_nbr = a # heavy atom's neighbor that isn't the acceptor
                    break

            pt1 = conf.GetAtomPosition(heavy_nbr_nbr.GetIdx())
            v1 = conf.GetAtomPosition(heavy_nbr.GetIdx())
            pt1 -= v1
            v1 -= cpt
            rotAxis = v1.CrossProduct(pt1)
            rotAxis.Normalize()
            bv1 = FeatDirUtils.ArbAxisRotation(120, rotAxis, v1)
            bv1.Normalize()
            bv2 = FeatDirUtils.ArbAxisRotation(-120, rotAxis, v1)
            bv2.Normalize()
            return [np.array(bv1), np.array(bv2)]

        elif len(nbrs) == 2: # sp3
            bvec = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
            bvec *= -1.0
            # we will create two vectors by rotating bvec by half the tetrahedral angle in either directions
            v1 = conf.GetAtomPosition(nbrs[0].GetIdx())
            v1 -= cpt
            v2 = conf.GetAtomPosition(nbrs[1].GetIdx())
            v2 -= cpt
            rotAxis = v1 - v2
            rotAxis.Normalize()
            bv1 = FeatDirUtils.ArbAxisRotation(54.5, rotAxis, bvec)
            bv2 = FeatDirUtils.ArbAxisRotation(-54.5, rotAxis, bvec)
            bv1.Normalize()
            bv2.Normalize()
            
            return [np.array(bv1), np.array(bv2)]
        
    else:  
        # take average direction of bonds and reverse it
        ave_bond = np.zeros(3)
        cnt = 0
        
        for nbor in rdmol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nbor_coords = np.array(rdmol.GetConformer().GetAtomPosition(nbor.GetIdx()))
            ave_bond += nbor_coords - atom_coords 
            cnt += 1
        
        if cnt > 0:
            ave_bond /= cnt
            ave_bond *= -1
            ave_bond = ave_bond / np.linalg.norm(ave_bond)
            return [ave_bond]
