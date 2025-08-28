import numpy as np
import rdkit.Chem as Chem
from scipy.spatial.distance import cdist
from omtra.data.pharmvec import GetDonorFeatVects, GetAcceptorFeatVects, GetAromaticFeatVects, GetHalogenFeatVects
from omtra.constants import ph_idx_to_type, ph_type_to_idx

#from line_profiler import profile

smarts_patterns = {
    'Aromatic': ["a1aaaaa1", "a1aaaa1"],
    'PositiveIon': ['[+,+2,+3,+4]', "[$(C(N)(N)=N)]", "[$(n1cc[nH]c1)]"],
    'NegativeIon': ['[-,-2,-3,-4]', "C(=O)[O-,OH,OX1]"],
    'HydrogenAcceptor': [
        "[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]", 
        "[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]"
    ],
    'HydrogenDonor': [
        "[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]", "[#8!H0&!$([OH][C,S,P]=O)]", "[#16!H0]"
    ],
    'Hydrophobic': [
    "a1aaaaa1", "a1aaaa1", 
    "[$([CH3X4,CH2X3,CH1X2])&!$(**[CH3X4,CH2X3,CH1X2])]",
    "[$(*([CH3X4,CH2X3,CH1X2])[CH3X4,CH2X3,CH1X2])&!$(*([CH3X4,CH2X3,CH1X2])([CH3X4,CH2X3,CH1X2])[CH3X4,CH2X3,CH1X2])]([CH3X4,CH2X3,CH1X2])[CH3X4,CH2X3,CH1X2]",
    "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2]",
    "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    "[$([S]~[#6])&!$(S~[!#6])]"
    ],
    'Halogen': [
        "[F;$(F-[#6]);!$(FC[F,Cl,Br,I])]",
        "[Cl;$(Cl-[#6]);!$(FC[F,Cl,Br,I])]",
        "[Br;$(Br-[#6]);!$(FC[F,Cl,Br,I])]",
        "[I;$(I-[#6]);!$(FC[F,Cl,Br,I])]"
    ]
}

matching_types = {
    'Aromatic': ['Aromatic', 'PositiveIon',],
    'HydrogenDonor': ['HydrogenAcceptor'],
    'HydrogenAcceptor': ['HydrogenDonor', 'Halogen'],
    'PositiveIon': ['NegativeIon', 'Aromatic'],
    'NegativeIon': ['PositiveIon'],
    'Hydrophobic': ['Hydrophobic', 'Halogen'],
    'Halogen': ['Hydrophobic', 'HydrogenAcceptor']
}

matching_distance = {
    "Aromatic": [7, 5],
    "HydrogenDonor": [4],
    "HydrogenAcceptor": [4, 4],
    "PositiveIon": [5, 5],
    "NegativeIon": [5],
    "Hydrophobic": [5, 4,],
    'Halogen': [5, 4]
}
#@profile
def get_smarts_matches(rdmol, smarts_pattern):
    """Find positions of a SMARTS pattern in molecule."""
    feature_positions = []
    atom_positions = []
    smarts_mol = Chem.MolFromSmarts(smarts_pattern)
    matches = rdmol.GetSubstructMatches(smarts_mol, uniquify=True)
    conformer = rdmol.GetConformer()
    positions = conformer.GetPositions()

    for match in matches:
        atoms = positions[list(match)]
        #atoms = [np.array(conformer.GetAtomPosition(idx)) for idx in match]
        feature_location = np.mean(atoms, axis=0)
        
        atom_positions.append(atoms)
        feature_positions.append(feature_location)

    return matches, atom_positions, feature_positions
#@profile
def get_vectors(mol, feature, atom_idxs, atom_positions, feature_positions):
    """Return direction vector(s) for all matches of a smarts pattern"""
    vectors = []
    for featAtoms, atomsLoc, featLoc in zip(atom_idxs, atom_positions, feature_positions):
        if feature == 'Aromatic':
            vec = GetAromaticFeatVects(atomsLoc, featLoc)
        elif feature == 'HydrogenDonor':
            vec = GetDonorFeatVects(featAtoms, atomsLoc, mol)
        elif feature == 'HydrogenAcceptor':
            vec = GetAcceptorFeatVects(featAtoms, atomsLoc, mol)
        elif feature == 'Halogen':
            vec = GetHalogenFeatVects(featAtoms, atomsLoc, mol)
        else:
            vec = None

        if vec:
            vectors.append(vec)
        else:
            vectors.append([np.zeros(3)])
    return vectors
#@profile
def check_interaction(all_ligand_positions, receptor, feature):
    """
    Check if the ligand features interact with a matching receptor features. 
    If true, update vector to point to receptor feature.
    """
    paired_features = matching_types[feature]
    feature_cutoff = matching_distance[feature]
    interaction = [False] * len(all_ligand_positions)
    updated_vectors = [None] * len(all_ligand_positions)

    for feature, cutoff in zip(paired_features, feature_cutoff):
        all_receptor_positions = []
        for rec_pattern in smarts_patterns[feature]:
            _, _, rec_feature_positions = get_smarts_matches(receptor, rec_pattern)
            all_receptor_positions.extend(rec_feature_positions)

        if not all_receptor_positions:
            continue
        
        all_receptor_positions = np.array(all_receptor_positions)
        distances = cdist(all_ligand_positions, all_receptor_positions)
        for i, dist in enumerate(distances):
            if np.any(dist <= cutoff):
                interaction[i] = True
                closest_receptor_position = all_receptor_positions[np.argmin(dist)]
                vector = closest_receptor_position - all_ligand_positions[i]
                vector = vector / np.linalg.norm(vector)
                updated_vectors[i] = [vector]

    return interaction, updated_vectors
#@profile
def get_pharmacophore_dict(ligand, receptor=None):
    """Extract pharmacophores and direction vectors from RDKit molecule.
        
    Returns
    -------
    dictionary : {'FeatureName' : {
                                   'P': [(coord), ... ],
                                   'V': [(vec), ... ],
                                   'I': [True/False, ...] # if receptor
                                   }
                 }
    """
    
    pharmacophores = {feature: {'P': [], 'V': [], 'I': []} for feature in smarts_patterns}
    
    Chem.SanitizeMol(ligand)
    ligand = Chem.AddHs(ligand, addCoords=True)
    if receptor: receptor = Chem.AddHs(receptor, addCoords=True)

    for feature, patterns in smarts_patterns.items():
        all_ligand_positions = []
        all_ligand_vectors = []
        
        for pattern in patterns:
            atom_idxs, atom_positions, feature_positions = get_smarts_matches(ligand, pattern)
            if feature_positions:
                vectors = get_vectors(ligand, feature, atom_idxs, atom_positions, feature_positions)
                all_ligand_positions.extend(feature_positions)
                all_ligand_vectors.extend(vectors)
        
        if all_ligand_positions:
            if receptor:
                interaction, updated_vectors = check_interaction(all_ligand_positions, receptor, feature)
                for i in range(len(all_ligand_vectors)):
                    if interaction[i]: 
                        all_ligand_vectors[i] = updated_vectors[i]

                pharmacophores[feature]['I'].extend(interaction)
                # receptor = Chem.RemoveHs(receptor)
            
            pharmacophores[feature]['P'].extend(all_ligand_positions)
            pharmacophores[feature]['V'].extend(all_ligand_vectors)            
    
    ligand = Chem.RemoveHs(ligand)
    # if receptor: receptor = Chem.RemoveHs(receptor)                
                
    return pharmacophores
#@profile
def get_pharmacophores(mol, rec=None):
    pharmacophores_dict = get_pharmacophore_dict(mol, rec) if rec else get_pharmacophore_dict(mol)
        
    X, P, V, I = [], [], [], []
    for ptype in pharmacophores_dict:
        pos = pharmacophores_dict[ptype]['P']
        P.extend(pos)
        V.extend(pharmacophores_dict[ptype]['V'])
        I.extend(pharmacophores_dict[ptype]['I'])
        type_embed = ph_type_to_idx[ptype]
        X.extend([type_embed]*len(pos))
        
    # V has shape (num_pharm_centers, num_vectors, 3)
    # where num_vectors is the maximum number of vectors for any pharmacophore center
    # num_vectors = max(len(v) for v in V)
    num_vectors=4
    for i in range(len(V)):
        if len(V[i]) > num_vectors:
            raise ValueError("pharmacophore center exceed max number of anticipated vectors")
        V[i].extend([np.zeros(3)] * (num_vectors - len(V[i])))

    P = np.array(P) # has shape (num_pharm_centers, 3)
    V = np.array(V) # has shape (num_pharm_centers, num_vectors, 3)
    X = np.array(X)

    if P.shape[0] == 0:
        P = np.zeros((0, 3))
        X = np.zeros(0)
        V = np.zeros((0, num_vectors, 3))

        if I != []:
            raise NotImplementedError("Didn't know how to handle this edge case for I because I was working on ligand-only data")

    return P, X, V, I