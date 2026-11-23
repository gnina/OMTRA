import unittest
import numpy as np
from omtra.eval.metrics.refinement import quality,rmsds,lddt_pli


class PocketQualityTests(unittest.TestCase):
    def setUp(self):
        self.xyz=np.array([[0.,0,0],[1.5,0,0],[1.5,1.5,0],[0,0,1.5]])

    def test_common_frame_retains_translation_and_rotation(self):
        moved=self.xyz @ np.array([[0.,-1,0],[1,0,0],[0,0,1]])+np.array([4.,-2,1])
        result=rmsds(moved,self.xyz)
        self.assertGreater(result['rmsd_common'],1)
        self.assertLess(result['rmsd_aligned'],1e-12)

    def test_alignment_does_not_allow_mirror_reflection(self):
        reflected=self.xyz.copy(); reflected[:,0]*=-1
        self.assertGreater(rmsds(reflected,self.xyz)['rmsd_aligned'],.1)

    def test_mismatched_atoms_and_nonfinite_coordinates_fail(self):
        with self.assertRaises(ValueError): rmsds(self.xyz[:-1],self.xyz)
        broken=self.xyz.copy(); broken[0,0]=np.nan
        with self.assertRaises(ValueError): rmsds(broken,self.xyz)

    def test_geometry_detects_bond_stretch_and_excludes_bonded_clashes(self):
        bonds=np.array([[0,1],[1,2],[0,3]])
        baseline=quality(self.xyz,self.xyz,bonds,[1.7]*4)
        self.assertEqual(baseline['bond_mae'],0)
        self.assertEqual(baseline['angle_mae'],0)
        self.assertEqual(baseline['nonbonded_pair_count'],1)
        broken=self.xyz.copy(); broken[1,0]+=2
        self.assertGreater(quality(broken,self.xyz,bonds,[1.7]*4)['bond_mae'],.5)

    def test_lddt_uses_joint_frame_and_checks_correspondence(self):
        ligand=np.array([[2.,2,2]])
        self.assertEqual(lddt_pli(self.xyz,ligand,self.xyz,ligand),1.)
        self.assertLess(lddt_pli(self.xyz+10,ligand,self.xyz,ligand),1.)
        with self.assertRaises(ValueError): lddt_pli(self.xyz[:-1],ligand,self.xyz,ligand)

if __name__=='__main__': unittest.main()
