import numpy as np
from rdkit import Chem


def get_chiral_centers_lite(mol, conf_id=-1):
    n_atoms = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol,addCoords=True)

    if mol_h.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers.")
    if conf_id >= mol_h.GetNumConformers():
        raise ValueError(f"conf_id {conf_id} out of range.")
    
    chiral_centers = np.zeros(mol_h.GetNumAtoms(), dtype=int)

    # --- R/S chirality ---
    try:
        Chem.AssignStereochemistryFrom3D(mol_h, confId=conf_id, replaceExistingTags=False)
        Chem.AssignCIPLabels(mol_h)

        for atom in mol_h.GetAtoms():
            if atom.HasProp('_CIPCode'):
                ctype = atom.GetProp('_CIPCode')
                chiral_centers[atom.GetIdx()] = 1 if ctype == 'R' else 2

    except Exception as e:
        print(f"Failed to compute R/S chirality: {e}", flush=True)

    # --- E/Z chirality ---
    try:
        Chem.AssignStereochemistry(mol_h, force=True, cleanIt=True)
        for bond in mol_h.GetBonds():
            st = bond.GetStereo()
            if st == Chem.BondStereo.STEREOE:
                chiral_centers[bond.GetBeginAtomIdx()] = 3
                chiral_centers[bond.GetEndAtomIdx()] = 3
            elif st == Chem.BondStereo.STEREOZ:
                chiral_centers[bond.GetBeginAtomIdx()] = 4
                chiral_centers[bond.GetEndAtomIdx()] = 4

    except Exception as e:
        print(f"Failed to compute E/Z chirality: {e}", flush=True)

    chiral_centers = chiral_centers[:n_atoms, None]

    return chiral_centers


def get_chiral_centers(mol_h, mol_idx, conf_id=-1):

    if mol_h.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers.")
    if conf_id >= mol_h.GetNumConformers():
        raise ValueError(f"conf_id {conf_id} out of range.")

    RS_centers = []
    EZ_centers = []
    failure_info = []

    # --- R/S chirality ---
    try:
        Chem.AssignStereochemistryFrom3D(mol_h, confId=conf_id, replaceExistingTags=True)
        Chem.AssignCIPLabels(mol_h)

        for atom in mol_h.GetAtoms():
            if atom.HasProp('_CIPCode'):
                RS_centers.append((str(atom.GetProp('_CIPCode')), int(atom.GetIdx())))

    except Exception as e:
        print(f"Failed to compute R/S chirality for {mol_idx}: {e}", flush=True)
        failure_info.append((mol_idx, 'RS', e))

    # --- E/Z chirality ---
    try:
        Chem.AssignStereochemistry(mol_h, force=True, cleanIt=True)
        for bond in mol_h.GetBonds():
            st = bond.GetStereo()
            if st == Chem.BondStereo.STEREOE:
                EZ_centers.append(('E', int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
            elif st == Chem.BondStereo.STEREOZ:
                EZ_centers.append(('Z', int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
    except Exception as e:
        print(f"Failed to compute E/Z chirality {mol_idx}: {e}", flush=True)
        failure_info.append((mol_idx, 'EZ', e))
   

    chiral_centers = np.zeros(mol_h.GetNumAtoms(), dtype=int)

    RS_edges = []
    for c_type, idx in RS_centers:
        chiral_centers[idx] = 1 if c_type == 'R' else 2
        c_atom = mol_h.GetAtomWithIdx(idx)
        neighbors = [nb for nb in c_atom.GetNeighbors()]
        
        if not neighbors:
            continue
        
        # sort by CIP rank
        neighbors.sort(key=lambda nb: int(nb.GetProp('_CIPRank')), reverse=True)
        RS_edges.append([sorted([c_atom.GetIdx(), nb.GetIdx()]) for nb in neighbors])

    EZ_edges = []
    for ctype, idx1, idx2 in EZ_centers:
        code = 3 if ctype == 'E' else 4
        chiral_centers[idx1] = chiral_centers[idx2] = code

        atom1 = mol_h.GetAtomWithIdx(idx1)
        atom2 = mol_h.GetAtomWithIdx(idx2)

        # get heavy-atom neighbors excluding the central bond
        nbs1 = [nb for nb in atom1.GetNeighbors() if nb.GetIdx() != idx2]
        nbs2 = [nb for nb in atom2.GetNeighbors() if nb.GetIdx() != idx1]

        # sort by CIPRank if available
        nbs1.sort(key=lambda nb: int(nb.GetProp('_CIPRank')) if nb.HasProp('_CIPRank') else 0, reverse=True)
        nbs2.sort(key=lambda nb: int(nb.GetProp('_CIPRank')) if nb.HasProp('_CIPRank') else 0, reverse=True)

        # take top 2 neighbors per atom (or fewer if not enough)
        edges1 = [sorted([idx1, nb.GetIdx()]) for nb in nbs1[:2]]
        edges2 = [sorted([idx2, nb.GetIdx()]) for nb in nbs2[:2]]

        EZ_edges.append([
            edges1,
            edges2,
        ])

    return RS_centers, RS_edges, EZ_centers, EZ_edges, chiral_centers, failure_info



def get_chiral_feats(mol, edge_index, mol_idx):
    n_atoms = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol,addCoords=True)

    RS_centers, RS_edges, EZ_centers, EZ_edges, chiral_centers, failure_info = get_chiral_centers(mol_h, mol_idx)
    chiral_centers = chiral_centers[:n_atoms, None]

    chiral_e_types = np.zeros((edge_index.shape[0],2), dtype=int)

    edge_map = {frozenset(e): i for i, e in enumerate(edge_index)}


    # Assign tetrahedral (R/S) chiral bond priorities
    for c_center, edges in zip(RS_centers, RS_edges):
        c_type, c_idx = c_center

        for priority, edge in enumerate(edges):
            i, j = edge
            idx = edge_map.get(frozenset(edge))
            if idx is not None:
                if i != c_idx:
                    # Bond goes TO the chiral center
                    chiral_e_types[idx, 0] = priority + 1
                elif j != c_idx:
                    # Bond originates FROM the chiral center
                    chiral_e_types[idx, 1] = priority + 1

    # Assign E/Z stereochemistry bonds
    for c_center, edges in zip(EZ_centers, EZ_edges):
        c_type, c_idx1, c_idx2 = c_center

        edges1 = edges[0]
        edges2 = edges[1]
        
        label = 5 if c_type == 'E' else 6

        central_idx = edge_map.get(frozenset(sorted([c_idx1, c_idx2])))
        if central_idx is not None:
            chiral_e_types[central_idx, 0] = label
            chiral_e_types[central_idx, 1] = label

        # Assign neighboring edges
        edge_idx = 7
        for edge in edges1:
            idx = edge_map.get(frozenset(edge))
            if idx is not None:
                i, j = edge
                if i != c_idx1 and i != c_idx2:
                    chiral_e_types[idx,0] = edge_idx
                elif j != c_idx1 and j != c_idx2:
                    chiral_e_types[idx,1] = edge_idx
            edge_idx+=1

        edge_idx = 7
        for edge in edges2:
            idx = edge_map.get(frozenset(edge))
            if idx is not None:
                i, j = edge
                if i != c_idx1 and i != c_idx2:
                    chiral_e_types[idx,0] = edge_idx
                elif j != c_idx1 and j != c_idx2:
                    chiral_e_types[idx,1] = edge_idx
            edge_idx+=1

    return chiral_e_types, chiral_centers, failure_info

def chirality_adj(chiral_centers, chiral_e_types, edge_index):
    n_atoms = chiral_centers.shape[0]
    chiral_adj = np.zeros((n_atoms, n_atoms), dtype=int)

    # Matrix of priority groups
    subs = -np.ones((n_atoms, 8), dtype=int)

    i, j = edge_index[:,0], edge_index[:,1]
    pi, pj = chiral_e_types[:, 0], chiral_e_types[:, 1]

    # Case 1: i is chiral center
    mask_i = (pi == 0) & (pj >= 1)
    subs[i[mask_i], pj[mask_i] - 1] = j[mask_i]

    # Case 2: j is chiral center
    mask_j = (pj == 0) & (pi >= 1) 
    subs[j[mask_j], pi[mask_j] - 1] = i[mask_j]

    # Collect R/S edges
    # –––––––––––––––––––––––––––––––
    rs_centers = np.where((chiral_centers == 1) | (chiral_centers == 2))[0]

    # Add edges
    # Directed cycle 1->2->3->1
    for center in rs_centers:
        p1, p2, p3, p4 = subs[center][:4]
        
        if p1 != -1 and p2 != -1:
            chiral_adj[p1, p2] = 1
        if p2 != -1 and p3 != -1:
            chiral_adj[p2, p3] = 1
        if p3 != -1 and p1 != -1:
            chiral_adj[p3, p1] = 1
        
        # Priority 4: bidirectional to 1,2,3
        if p4 != -1:
            for p in [p1, p2, p3]:
                if p != -1:
                    chiral_adj[p4, p] = 2
                    chiral_adj[p, p4] = 2

    # Collect E/Z edges
    # –––––––––––––––––––––––––––––––
    ez_centers = np.where((chiral_centers == 3) | (chiral_centers == 4))[0]

    for center in ez_centers:
        # Find the double bond atom pair from edge_index / chiral_e_types
        # chiral_e_types should have label > 4 for E/Z central bond (e.g., 5/6)
        bond_mask = ((i == center) | (j == center)) & (chiral_e_types[:,0] >= 5)
        bond_idx = np.where(bond_mask)[0]
        if len(bond_idx) == 0:
            continue
        idx = bond_idx[0]
        a, b = i[idx], j[idx]
            
        # Get top 2 neighbors on each side of EZ bond (priority 1 and 2)
        p_a1, p_a2 = subs[a, 6], subs[a, 7]
        p_b1, p_b2 = subs[b, 6], subs[b, 7]
        
        pairs = []
        if chiral_centers[center] == 3:  # E center
            pairs.extend([
                ((p_a1, p_b1), 3),  # E
                ((p_a2, p_b2), 3),  # E
                ((p_a1, p_b2), 4),  # Z
                ((p_a2, p_b1), 4)   # Z
            ])
        else:  # Z center
            pairs.extend([
                ((p_a1, p_b2), 3),  # E
                ((p_a2, p_b1), 3),  # E
                ((p_a1, p_b1), 4),  # Z
                ((p_a2, p_b2), 4)   # Z
            ])

        # Assign edges symmetrically
        for (u, v), val in pairs:
            if u != -1 and v != -1:
                chiral_adj[u, v] = val
                chiral_adj[v, u] = val

    return chiral_adj
