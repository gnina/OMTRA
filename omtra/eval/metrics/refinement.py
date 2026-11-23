"""Reference-relative pocket diagnostics; no training imports or gradients."""
from itertools import combinations
import numpy as np


def coordinates(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite (N, 3) coordinates')
    return x


def rmsds(pred, ref):
    pred, ref = coordinates(pred), coordinates(ref)
    if pred.shape != ref.shape:
        raise ValueError('Atom correspondence must be complete, with identical ordering')
    common = float(np.sqrt(np.mean(np.sum((pred-ref)**2, axis=1))))
    a, b = pred-pred.mean(0), ref-ref.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    aligned = float(np.sqrt(np.mean(np.sum((a @ (u @ correction @ vt)-b)**2, axis=1))))
    return {'rmsd_common': common, 'rmsd_aligned': aligned}


def topology(bonds, n):
    neighbors = [set() for _ in range(n)]
    for a, b in bonds:
        neighbors[int(a)].add(int(b)); neighbors[int(b)].add(int(a))
    angles = [(a,b,c) for b, ns in enumerate(neighbors) for a,c in combinations(sorted(ns),2)]
    excluded = np.eye(n, dtype=bool)
    for a, ns in enumerate(neighbors):
        for b in ns:
            excluded[a,b] = True
            for c in neighbors[b]: excluded[a,c] = True
    return np.asarray(angles, dtype=int).reshape(-1,3), np.triu(~excluded,1)


def angle_values(x, angles):
    a,b,c = angles.T
    v,w = x[a]-x[b],x[c]-x[b]
    den = np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1)
    if np.any(den < 1e-10): raise ValueError('Degenerate bond angle')
    return np.degrees(np.arccos(np.clip(np.sum(v*w,axis=1)/den,-1,1)))


def quality(pred, ref, bonds, radii):
    pred, ref = coordinates(pred), coordinates(ref)
    out = rmsds(pred,ref)
    bonds = np.asarray(bonds,dtype=int).reshape(-1,2)
    angles,nonbonded = topology(bonds,len(ref))
    for name,idx,fn,threshold in [
        ('bond',bonds,lambda x, ix: np.linalg.norm(x[ix[:,0]]-x[ix[:,1]],axis=1),.2),
        ('angle',angles,angle_values,20.)]:
        delta = np.abs(fn(pred,idx)-fn(ref,idx)) if len(idx) else np.array([])
        out[name+'_count'] = len(idx)
        out[name+'_mae'] = float(delta.mean()) if len(idx) else np.nan
        out[name+'_deviation_fraction'] = float((delta>threshold).mean()) if len(idx) else np.nan
    distances = np.linalg.norm(pred[:,None,:]-pred[None,:,:],axis=-1)
    radii = np.asarray(radii)
    clashes = nonbonded & (distances < radii[:,None]+radii[None,:]-.4)
    out['nonbonded_pair_count'] = int(nonbonded.sum())
    out['clash_pair_count'] = int(clashes.sum())
    out['clash_pairs_per_100_atoms'] = float(100*clashes.sum()/len(ref))
    return out


def lddt_pli(protein, ligand, ref_protein, ref_ligand):
    protein, ligand = coordinates(protein), coordinates(ligand)
    ref_protein,ref_ligand = coordinates(ref_protein),coordinates(ref_ligand)
    if protein.shape!=ref_protein.shape or ligand.shape!=ref_ligand.shape:
        raise ValueError('lDDT requires complete corresponding atoms')
    dref = np.linalg.norm(ref_protein[:,None]-ref_ligand[None,:],axis=-1)
    pi,li = np.where(dref<=6.)
    if not len(pi): return np.nan
    delta = np.abs(np.linalg.norm(protein[pi]-ligand[li],axis=-1)-dref[pi,li])
    return float(np.mean([np.mean(delta<=t) for t in (.5,1.,2.,4.)]))
