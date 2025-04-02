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
- [ ] npnde edges? don't want fully connected self-edges, only bonds may be insufficient
- [ ] we don't create node output heads for pharm vec features
- [ ] residue type as node feature?
- [ ] add covalent edge types to omtra/data/graph/__init__.py
- [ ] remove edges with n_categories=0 from moadlities