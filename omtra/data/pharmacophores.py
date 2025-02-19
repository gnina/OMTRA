import numpy as np
import rdkit.Chem as Chem
from scipy.spatial.distance import cdist
from pharmvec import GetDonorFeatVects, GetAcceptorFeatVects, GetAromaticFeatVects
import pprint

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
        "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
        "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
        "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
        "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
        "[$([S]~[#6])&!$(S~[!#6])]"
    ]
}

matching_types = {
    'Aromatic': ['Aromatic', 'PositiveIon'],
    'HydrogenDonor': ['HydrogenAcceptor'],
    'HydrogenAcceptor': ['HydrogenDonor'],
    'PositiveIon': ['NegativeIon', 'Aromatic'],
    'NegativeIon': ['PositiveIon'],
    'Hydrophobic': ['Hydrophobic']
}

matching_distance = {
    "Aromatic": [7, 5],
    "HydrogenDonor": [4],
    "HydrogenAcceptor": [4],
    "PositiveIon": [5, 7],
    "NegativeIon": [5],
    "Hydrophobic": [5]
}

def get_smarts_matches(rdmol, smarts_pattern):
    """Find positions of a SMARTS pattern in molecule."""
    feature_positions = []
    atom_positions = []
    smarts_mol = Chem.MolFromSmarts(smarts_pattern)
    matches = rdmol.GetSubstructMatches(smarts_mol, uniquify=True)
    for match in matches:
        atoms = [np.array(rdmol.GetConformer().GetAtomPosition(idx)) for idx in match]
        feature_location = np.mean(atoms, axis=0)
        
        atom_positions.append(atoms)
        feature_positions.append(feature_location)

    return matches, atom_positions, feature_positions

# get the vectors for all matches of a smarts pattern
def get_vectors(mol, feature, atom_idxs, atom_positions, feature_positions):
    vectors = []
    for featAtoms, atomsLoc, featLoc in zip(atom_idxs, atom_positions, feature_positions):
        if feature == 'Aromatic':
            vectors.append(GetAromaticFeatVects(atomsLoc, featLoc, scale=1.0))
        elif feature == 'HydrogenDonor':
            vectors.append(GetDonorFeatVects(featAtoms, atomsLoc, mol))
        elif feature == 'HydrogenAcceptor':
            vectors.append(GetAcceptorFeatVects(featAtoms, atomsLoc, mol))
        else:
            vectors.append(np.zeros(3))
    return vectors

def check_interaction(all_ligand_positions, receptor, feature):
    """
    Check if the ligand features interact with their matching receptor features
    """
    paired_features = matching_types[feature]
    feature_cutoff = matching_distance[feature]
    interaction = [False] * len(all_ligand_positions)

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

    return interaction

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
            pharmacophores[feature]['P'].extend(all_ligand_positions)
            pharmacophores[feature]['V'].extend(all_ligand_vectors)
            
            if receptor:
                interaction = check_interaction(all_ligand_positions, receptor, feature)
                pharmacophores[feature]['I'].extend(interaction)
                
    return pharmacophores

def get_pharmacophores(mol, ph_type_to_idx):
    pharmacophores_dict = get_pharmacophore_dict(mol)
    X, A, V = [], [], []
    for type in pharmacophores_dict:
        pos = pharmacophores_dict[type]['P']
        A.extend(pos)
        V.extend(pharmacophores_dict[type]['V'])
        type_embed = ph_type_to_idx[type]
        X.extend([type_embed]*len(pos))

    return X, A, V