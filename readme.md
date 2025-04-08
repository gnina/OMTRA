# One Model to Rule Them All

A multi-task generative model for small-molecule structure-based drug design. 

# Building the Environment

For now:
```bash
git clone https://github.com/gnina/OMTRA.git
cd OMTRA
mamba create -n omtra python=3.11
mamba activate omtra
chmod +x build_env.sh
./build_env.sh
```

# TODO:
- [ ] residue type as node feature?
- [ ] need to apply masking on node vec feature loss
- [ ] need to consider permutation invariance for vector feature prediction?
- [ ] CCD code frequency weighting in plinder dataset
- [ ] sampling code
- [ ] sampled system object, ligand builder object
- [ ] need to remove edges in vector field forward pass
- [ ] radius graph n edges approximation
- [ ] don't do fully connected NPNDE
- [ ] task ExpApoConditionedLigandDocking has npnde_x prior as gaussian -> needs to be apo_exp prior, but don't know if plinder dataset class will support this (PlinderDataset.get_link_coords)
    - turns out linked structures do not contain NPNDEsthe 
