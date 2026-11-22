import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from omtra.data.stereo_adjudication import compare_with_coordinates, classify, tautomer_blind_ties


def embedded(smiles, seed=5):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return Chem.RemoveHs(mol)


class AdjudicationTests(unittest.TestCase):
    def test_same_isomer_agrees_and_mirror_is_a_real_disagreement(self):
        for smiles in ('C[C@H](O)C(=O)O', 'C[C@H]1CCCC[C@@H]1O', 'F/C=C/[C@H](O)C'):
            mol = embedded(smiles)
            xyz = mol.GetConformer().GetPositions()
            self.assertEqual(classify(compare_with_coordinates(Chem.MolFromSmiles(smiles), xyz)), 'agree', smiles)
            result = classify(compare_with_coordinates(Chem.MolFromSmiles(smiles), xyz * np.array([-1.0, 1.0, 1.0])))
            self.assertEqual(result, 'real', smiles)   # the carbon centre inverts under reflection (E/Z does not)

    def test_equivalent_phosphate_oxygens_are_pseudo_and_real_phosphorus_is_not(self):
        phosphate = Chem.MolFromSmiles('C[P@](=O)(O)OC')           # =O and OH equivalent up to protonation
        self.assertTrue(tautomer_blind_ties(phosphate, 1))
        thiophosphate = Chem.MolFromSmiles('C[P@](=O)(S)OC')       # O and S are different elements: a real centre
        self.assertFalse(tautomer_blind_ties(thiophosphate, 1))

    def test_carbon_centres_with_vinyl_or_aldehyde_branches_are_never_pseudo(self):
        # Bond order / H count must not make genuinely different carbon substituents "tied".
        for smiles in ('C[C@H](CC)C=C', 'C[C@H](C=O)CO', 'CC[C@H](C)[C@H](N)C(=O)O'):
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                    self.assertFalse(tautomer_blind_ties(mol, atom.GetIdx()), (smiles, atom.GetIdx()))

    def test_pseudo_only_disagreement_is_not_counted_as_real(self):
        mol = embedded('CO[P@](=O)(O)OC')
        xyz = mol.GetConformer().GetPositions()
        opposite = Chem.MolFromSmiles('CO[P@@](=O)(O)OC')
        comparison = compare_with_coordinates(opposite, xyz)
        # whichever way the arbitrary tag falls, the phosphate centre must never be classed as a real disagreement
        self.assertIn(classify(comparison), ('agree', 'pseudo_only'))
        self.assertTrue(all(a['pseudo'] for a in comparison['atoms']))

    def test_inverted_carbon_with_phosphate_present_is_real(self):
        smiles = 'C[C@H](O)COP(=O)(O)O'
        mol = embedded(smiles)
        xyz = mol.GetConformer().GetPositions()
        inverted = Chem.MolFromSmiles('C[C@@H](O)COP(=O)(O)O')
        self.assertEqual(classify(compare_with_coordinates(inverted, xyz)), 'real')


if __name__ == '__main__':
    unittest.main()
