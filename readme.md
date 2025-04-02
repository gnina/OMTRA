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
<<<<<<< HEAD
- [ ] npnde edges? don't want fully connected self-edges, only bonds may be insufficient
- [ ] node output heads for pharm vec features!!!!!!!!!
- [ ] residue type as node feature?
- [ ] add covalent edge types to omtra/data/graph/__init__.py
- [ ] remove edges with n_categories=0 from moadlities
=======
- [ ] how to model npnde? need to maintain edge features? writing npndes back out requires we have edges? but we don't really need to model npnde edges?
- [ ] node output head for vec features needs to be handled appropriately
- [ ] how does heterogvpconv create messaging and update functions?
- [ ] why do we have modalities with n_categories=0?
- [ ] need to apply masking on node vec feature loss
- [ ] need to consider permutation invariance for vector feature prediction?
>>>>>>> c78f03b2b34084b1d6d8634b38acbeecec6a0f46
