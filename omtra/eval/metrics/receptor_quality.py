"""Noisy-holo refinement diagnostics in the supplied common coordinate frame."""
import numpy as np
from rdkit import Chem


def receptor_quality(system):
    if 'protein_structure' not in system.task.groups_generated:return {}
    from scripts.refinement_eval.metrics import quality,rmsds
    g=system.g;data=g.nodes['prot_atom'].data
    if any(k not in data for k in ('x_0','x_1','x_1_true')):
        return {'receptor_metrics_present':False}
    reference=system.get_protein_array(reference=True,include_bonds=True)
    true=data['x_1_true'].cpu().numpy();initial=data['x_0'].cpu().numpy();output=data['x_1'].cpu().numpy()
    bonds=reference.bonds.as_array()[:,:2] if reference.bonds is not None else np.empty((0,2),dtype=int)
    result={}
    try:
        table=Chem.GetPeriodicTable();radii=np.asarray([table.GetRvdw(str(e).capitalize()) for e in reference.element])      # PDB element symbols are upper case ('SE'); RDKit wants 'Se'
        for name,coords in [('initial',initial),('output',output)]:
            result.update({f'receptor_{name}_{k}':v for k,v in quality(coords,true,bonds,radii).items()})
            for subset,mask in [('backbone',np.isin(reference.atom_name,['N','CA','C','O','OXT'])),
                                ('sidechain',~np.isin(reference.atom_name,['N','CA','C','O','OXT']))]:
                if mask.any():result[f'receptor_{name}_{subset}_rmsd_common']=rmsds(coords[mask],true[mask])['rmsd_common']
            if len(bonds):
                a,b=bonds.T
                peptide=(((reference.atom_name[a]=='C')&(reference.atom_name[b]=='N'))|
                         ((reference.atom_name[a]=='N')&(reference.atom_name[b]=='C')))&(reference.res_id[a]!=reference.res_id[b])&(reference.chain_id[a]==reference.chain_id[b])
                aa,bb=a[peptide],b[peptide]
                result['receptor_peptide_bond_count']=len(aa)
                result[f'receptor_{name}_peptide_bond_mae']=float(np.abs(np.linalg.norm(coords[aa]-coords[bb],axis=1)-np.linalg.norm(true[aa]-true[bb],axis=1)).mean()) if len(aa) else None
        result['receptor_output_drift_rmsd_common']=rmsds(output,initial)['rmsd_common']
        if g.num_nodes('npnde'):
            nd=g.nodes['npnde'].data
            for name,suffix in [('initial','0'),('output','1')]:
                if 'x_'+suffix in nd:result[f'receptor_{name}_cofactor_rmsd_common']=rmsds(nd['x_'+suffix].cpu().numpy(),nd['x_1_true'].cpu().numpy())['rmsd_common']
    except (ValueError,RuntimeError):
        result['receptor_metrics_present']=False
        return result
    result['receptor_metrics_present']=True
    return result
