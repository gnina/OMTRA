"""Ligand-only chemistry, conformer recovery and requested-stereo diagnostics."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from omtra.data.requested_stereo import labels_from_identity


def ligand_quality(mol, reference=None):
    if mol is None:
        return {'canonical_smiles':None, 'n_heavy_atoms':None, 'conformer_rmsd':None}
    mol=Chem.Mol(mol)
    if mol.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(mol,replaceExistingTags=True)
    result={'canonical_smiles':Chem.MolToSmiles(mol,isomericSmiles=True),
            'n_heavy_atoms':mol.GetNumHeavyAtoms(),'n_components':len(Chem.GetMolFrags(mol)),
            'n_aromatic_rings':sum(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                                   for ring in mol.GetRingInfo().AtomRings()),
            'n_stereocentres':len(Chem.FindMolChiralCenters(mol,includeUnassigned=True)),
            'conformer_rmsd':None}
    if reference is not None and mol.GetNumAtoms()==reference.GetNumAtoms():
        try:
            result['conformer_rmsd']=float(rdMolAlign.GetBestRMS(Chem.Mol(mol),reference))
        except (RuntimeError, ValueError):
            pass
    return result


def requested_stereo_quality(system):
    from omtra.data.requested_stereo import eligible
    if not eligible(system.task) or 'stereo_atom' not in system.g.nodes['lig'].data:
        return {}
    expected=system.g.nodes['lig'].data['stereo_atom'].cpu().numpy()
    mask=np.isin(expected,[2,3,4,5])
    n=int(mask.sum())
    src,dst=system.g.edges(etype='lig_to_lig')
    requested_bonds=[]
    for a,b,label in zip(src.cpu().tolist(),dst.cpu().tolist(),system.g.edges['lig_to_lig'].data['stereo_bond'].cpu().tolist()):
        if a < b and label in (2,3):
            n+=1
            requested_bonds.append((a,b,label))
    if not n:return {'n_requested_stereo':0,'requested_stereo_match':None,'frac_requested_stereo_correct':None}
    failed={'n_requested_stereo':n,'requested_stereo_match':False,'frac_requested_stereo_correct':0.}
    try:
        mol=system.get_rdkit_ligand()
        if mol is None:return failed
        mol=Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol,replaceExistingTags=True)
        atom,bond=labels_from_identity(mol)
    except (RuntimeError,ValueError):return failed
    if len(atom)!=len(expected):return failed
    correct=int((atom[mask]==expected[mask]).sum())
    correct+=sum(int(bond.get((a,b),0)==label) for a,b,label in requested_bonds)
    return {'n_requested_stereo':n,'requested_stereo_match':bool(correct==n) if n else None,
            'frac_requested_stereo_correct':correct/n if n else None}
