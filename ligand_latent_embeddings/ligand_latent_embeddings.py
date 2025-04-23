import zarr
import numpy as np
import math

import torch
from torch import nn, einsum
import torch.utils.data as torch_data
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import dgl

import matplotlib.pyplot as plt

from omtra.models.gvp import HeteroGVPConv, _rbf, _norm_no_nan
from omtra.data.xace_ligand import sparse_to_dense
from omtra.constants import lig_atom_type_map, charge_map




class ZarrDataset(torch_data.Dataset):
    def __init__(self, zarr_store):
        self.zarr_store = zarr_store
        self.n_graphs = self.zarr_store['lig/node/graph_lookup'].shape[0]

    def __len__(self):
        return self.n_graphs
    
    def __getitem__(self, idx):

        # get node and edge data groups from zarr store
        node_data = self.zarr_store['lig/node']
        edge_data = self.zarr_store['lig/edge']

        # lookup start and end indicies for node and edge data to pull just one graph from the full dataset
        node_start_idx, node_end_idx = node_data['graph_lookup'][idx]
        edge_start_idx, edge_end_idx = edge_data['graph_lookup'][idx]

        # Pull out data for the graph
        x = node_data['x'][node_start_idx:node_end_idx]
        a = node_data['a'][node_start_idx:node_end_idx]
        c = node_data['c'][node_start_idx:node_end_idx]
        e = edge_data['e'][edge_start_idx:edge_end_idx]
        edge_idxs = edge_data['edge_index'][edge_start_idx:edge_end_idx]

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        a_tensor = torch.tensor(a, dtype=torch.int32)
        c_tensor = torch.tensor(c, dtype=torch.int32)
        e_tensor = torch.tensor(e, dtype=torch.int32)

        # Convert to dense representation
        x_tensor, a_tensor, c_tensor, e_tensor, edge_idxs = sparse_to_dense(x_tensor, a_tensor, c_tensor, e_tensor, torch.tensor(edge_idxs, dtype=torch.int32))  

        # Create heterogeneous graph
        g = dgl.heterograph({('lig', 'lig_to_lig', 'lig'): (edge_idxs[0, :], edge_idxs[1, :])}, num_nodes_dict={'lig': node_end_idx - node_start_idx})

        g.nodes['lig'].data['a_1_true'] = a_tensor
        g.nodes['lig'].data['c_1_true'] = c_tensor
        g.nodes['lig'].data['x_1_true'] = x_tensor
        g.edges['lig_to_lig'].data['e_1_true'] = e_tensor.unsqueeze(1)

        return g
    


class Encoder(nn.Module):
    def __init__(self, node_types, edge_types, scalar_size, vector_size, num_gvp_layers, mlp_hidden_size, embedding_dim, rbf_dim, rbf_dmax):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(scalar_size, mlp_hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(mlp_hidden_size, embedding_dim))
        
        self.gvps = nn.ModuleList([HeteroGVPConv(node_types= node_types,
                                                edge_types= edge_types,
                                                scalar_size= embedding_dim,
                                                vector_size= vector_size,
                                                use_dst_feats= False,
                                                rbf_dim = rbf_dim,
                                                rbf_dmax= rbf_dmax
                                                ) for i in range(num_gvp_layers)])   
        

    def forward(self, g, scalar_feats, coord_feats, vector_feats, edge_feats, x_diff, d):
        
        # Pass to MLP to change dimension 
        scalar_feats_reshaped = self.mlp(scalar_feats)  # (n_atoms, embedding_dim)
        scalar_feats_reshaped = {'lig': scalar_feats_reshaped}

        # Sequentially pass through HeteroGVPConv layers
        for gvp_layer in self.gvps:
            scalar_feats_reshaped, vector_feats = gvp_layer(g= g,
                               scalar_feats= scalar_feats_reshaped,
                               coord_feats= coord_feats,
                               vec_feats= vector_feats,
                               edge_feats= edge_feats,
                               x_diff= x_diff,
                               d= d
                               )
            
        return scalar_feats_reshaped['lig']



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost


    def forward(self, z_e):
        # z_e = (n_atoms, embedding_dim)
        # embedding = (num_embeddings, embedding_dim)
        distances = (torch.sum(z_e**2, dim=1, keepdim=True)         # (num_atoms, num_embeddings)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_e, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # Get indices of closest codebook (embedding) vector
        quantized = self.embedding(encoding_indices).squeeze(1)   # (num_atoms, embedding_dim)
        
        # Loss
        commitment_loss = F.mse_loss(quantized.detach(), z_e)
        vq_loss = F.mse_loss(quantized, z_e.detach())
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Perplexity 
        quantized = z_e + (quantized - z_e).detach()
        encodings = F.one_hot(encoding_indices, num_classes=self.embedding.num_embeddings).squeeze(1).float()
        avg_probs = torch.mean(encodings, dim=0)  
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))    #  "spread" of the quantized embeddings. Indicates how well the codebook is being used 
        
        return loss, quantized, perplexity
    


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_decod_hiddens, num_decod_layers, num_bond_decod_hiddens, num_bond_decod_layers, num_atom_types, num_atom_charges, num_bond_orders):
        super(Decoder, self).__init__()
        
        # Decoder for reconstructing atom type and charge logits from latent vector
        decod_layers = [nn.Linear(embedding_dim, num_decod_hiddens), nn.ReLU()]  # Input layer + activation
        
        # Add hidden layers
        for _ in range(num_decod_layers):
            decod_layers.append(nn.Linear(num_decod_hiddens, num_decod_hiddens))
            decod_layers.append(nn.ReLU())  # Non-linearity between layers
        
        # Output layer
        decod_layers.append(nn.Linear(num_decod_hiddens, num_atom_types+num_atom_charges))

        # Use nn.Sequential to define the entire model
        self.decoder = nn.Sequential(*decod_layers)


        # MLP to predict bond order from atom type and charge logits
        bond_decod_layers = [
            nn.Linear(embedding_dim+1, num_bond_decod_hiddens),
            nn.ReLU()
        ]
        
        # Add hidden layers
        for _ in range(num_bond_decod_layers):
            bond_decod_layers.append(nn.Linear(num_bond_decod_hiddens, num_bond_decod_hiddens))
            bond_decod_layers.append(nn.ReLU())
        
        # Ouptut layer
        bond_decod_layers.append(nn.Linear(num_bond_decod_hiddens, num_bond_orders))
        
        # Use nn.Sequential to define the entire model
        self.bond_decoder = nn.Sequential(*bond_decod_layers)

    

    def forward(self, quantized, dists, pair_indices):

        scalar_feats_logits = self.decoder(quantized)  # (num_atoms, num_atom_types + num_atom_charges)

        scalar_feats_0 = quantized[pair_indices[:, 0]]
        scalar_feats_1 = quantized[pair_indices[:, 1]]

        combined_quantized_dists = torch.concat([scalar_feats_0+scalar_feats_1, dists], dim=1) # (P, num_atom_types + num_atom_charges + 1)
        
        bond_order_logits = self.bond_decoder(combined_quantized_dists)    # (P, num_bond_orders)
       
        return scalar_feats_logits, bond_order_logits
    


class Model(nn.Module):
    def __init__(self,
                 num_atom_types, 
                 num_atom_charges,
                 num_bond_orders,
                 vector_size,
                 num_gvp_layers,
                 mlp_hidden_size,
                 embedding_dim, 
                 num_embeddings, 
                 num_decod_hiddens, 
                 num_decod_layers, 
                 num_bond_decod_hiddens, 
                 num_bond_decod_layers, 
                 commitment_cost,
                 rbf_dim=32,
                 rbf_dmax=10):
                 
        super(Model, self).__init__()
        self.num_atom_types = num_atom_types
        self.num_atom_charges = num_atom_charges
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.vector_size = vector_size
        
        self.encoder = Encoder(node_types = ['lig'], # Only ligand nodes
                               edge_types = ['lig_to_lig'], # Only ligand edges
                               scalar_size= self.num_atom_types + self.num_atom_charges, 
                               vector_size = vector_size,
                               num_gvp_layers= num_gvp_layers, 
                               mlp_hidden_size= mlp_hidden_size,
                               embedding_dim=  embedding_dim,
                               rbf_dim = self.rbf_dim,
                               rbf_dmax = self.rbf_dmax)

        self.vq_vae = VectorQuantizer(num_embeddings = num_embeddings,
                                      embedding_dim = embedding_dim,
                                      commitment_cost = commitment_cost)
       
        self.decoder = Decoder(embedding_dim = embedding_dim,
                               num_decod_hiddens = num_decod_hiddens,
                               num_decod_layers = num_decod_layers,
                               num_bond_decod_hiddens = num_bond_decod_hiddens,
                               num_bond_decod_layers = num_bond_decod_layers,
                               num_atom_types = self.num_atom_types,
                               num_atom_charges = self.num_atom_charges,
                               num_bond_orders = num_bond_orders
                               )


    def forward(self, g):

        """ Get relevant features from batched graph """
        # Get node atom types from graph and one-hot encode
        atom_types = g.nodes['lig'].data['a_1_true']    
        one_hot_atom_types = F.one_hot(atom_types.to(torch.long), num_classes=self.num_atom_types).float()

        # Get node atom charges from graph and one-hot encode
        atom_charges = g.nodes['lig'].data['c_1_true']  
        shifted_charges = atom_charges + 5      # TODO: Remove. Will be fixed by data loader
        one_hot_atom_charges = F.one_hot(shifted_charges.to(torch.long), num_classes=self.num_atom_charges).float()

        scalar_feats = torch.cat([one_hot_atom_types, one_hot_atom_charges], dim=1) # (n_atoms, n_atom_types + n_atom_charges)

        vector_feats = {'lig': torch.zeros((scalar_feats.shape[0], self.vector_size, 3), dtype=torch.float32)}    # Vector features
        coord_feats = {'lig': g.nodes['lig'].data['x_1_true']}   # Atom coordinates
        edge_feats = {'lig_to_lig': g.edges['lig_to_lig'].data['e_1_true'].unsqueeze(1)} # Bond order
        
        edges = g.edges(etype='lig_to_lig')
        diff = coord_feats['lig'][edges[0]] - coord_feats['lig'][edges[1]]
        d = _norm_no_nan(diff)  # (num_edges,)
        x_diff = diff / d.unsqueeze(1)   # (num_edges,)

        d_rbf = _rbf(d, D_min=0, D_max=self.rbf_dmax, D_count=self.rbf_dim) # _rbf:  D_min=0 and D_max=10, D_count=32
        d = {'lig_to_lig': d_rbf}   # Convert to tensor dictionary
        x_diff = {'lig_to_lig': x_diff} # Convert to tensor dictionary

       
        
        """ Indexing into batched graph to get pairwise atom distances """

        lig_features = g.nodes['lig'].data['x_1_true'] # (N, d) TODO: no 's' in my graph. change to match dataloader

        # Get the number of 'lig' nodes in each graph component (this is a list or tensor)
        num_nodes_list = g.batch_num_nodes('lig')

        # Convert the list to a tensor (ensure it is on the same device as lig_features)
        num_nodes_tensor = num_nodes_list.clone().detach().to(device=lig_features.device)

        # Precompute the offsets by using torch.cumsum.
        # The offset for each graph is the cumulative sum of nodes in previous graphs.
        # For instance, if num_nodes_tensor = [n1, n2, n3, ...],
        # then offsets = [0, n1, n1+n2, ...].
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=lig_features.device),
            num_nodes_tensor.cumsum(dim=0)[:-1]
        ])

        pair_indices_list = []  # to accumulate indices for each subgraph

        # Zip together the precomputed offset with the corresponding number of nodes.
        for offset, num_nodes in zip(offsets, num_nodes_list):
            # Only consider graphs with at least 2 nodes
            if num_nodes < 2:
                continue

            # Get unique pair indices within the current subgraph.
            # This returns a tensor of shape [2, num_pairs] where the first row is the
            # row indices and the second row is the column indices (with row < col).
            local_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
            
            # Adjust the local indices to be global in the batched graph by adding the offset.
            global_indices = local_indices + offset
            pair_indices_list.append(global_indices)

        # Concatenate the indices from all subgraphs along the pair (second) dimension.
        if pair_indices_list:
            # pair_indices shape will be [2, total_num_pairs]
            pair_indices = torch.cat(pair_indices_list, dim=1)
            
            # Transpose to shape [total_num_pairs, 2] so that each row is a pair of indices.
            pair_indices = pair_indices.t()  # shape: [P, 2]
        else:
            pair_indices = torch.empty((0, 2), dtype=torch.long, device=lig_features.device)


        # Get corresponding set of bond orders
        bond_orders = g.edges['lig_to_lig'].data['e_1_true'].unsqueeze(1).view(-1)
        src, dest = g.edges(etype='lig_to_lig')

        # Create a lookup matrix of size [N, N] where N is total number of lig nodes
        N = g.num_nodes('lig')
        lookup = torch.zeros((N, N), dtype=torch.int64, device=g.device)
        lookup[src, dest] = bond_orders
        lookup[dest, src] = bond_orders

        bond_orders_for_pairs = lookup[pair_indices[:, 0], pair_indices[:, 1]].unsqueeze(1)

        # Get the positions of each atom in the pair
        x_i = coord_feats['lig'][pair_indices[:, 0]]  # (P, 3)
        x_j = coord_feats['lig'][pair_indices[:, 1]]  # (P, 3)

        # Euclidean distance
        dists = _norm_no_nan(x_i - x_j).unsqueeze(1)  # (P, 1)



        """ Pass to VQ-VAE model """
        z_e = self.encoder(g, scalar_feats, coord_feats, vector_feats, edge_feats, x_diff, d)   # Encoding

        loss, z_d, perplexity = self.vq_vae(z_e)    # Vector Quantization
        
        scalar_feats_logits, bond_order_logits = self.decoder(z_d, dists, pair_indices)    # Decoding
        atom_type_logits = scalar_feats_logits[:, :self.num_atom_types]    # (num_atoms, num_atom_types)
        atom_charge_logits = scalar_feats_logits[:, self.num_atom_types:]  # (num_atoms, num_atom_charges)


        recon_loss = F.cross_entropy(atom_type_logits, atom_types.long()) + F.cross_entropy(atom_charge_logits, shifted_charges.long()) + F.cross_entropy(bond_order_logits, bond_orders_for_pairs.squeeze().long())   # Reconstruction loss
        loss = loss + recon_loss

        return loss, atom_type_logits, atom_charge_logits, bond_order_logits, perplexity


def display_graph(g):
    """" Displays node and edge feature shapes for a dgl graph """

    # Print node features
    print("\nNode Features:")
    print("x:", g.nodes['lig'].data['x_1_true'].shape)
    print("a:", g.nodes['lig'].data['a_1_true'].shape)
    print("c", g.nodes['lig'].data['c_1_true'].shape)
    print("e:", g.edges['lig_to_lig'].data['e_1_true'].shape)
    print(g.edges['lig_to_lig'].data['e_1_true'].dtype)


""" Data Class """
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import os
from omtra.utils import omtra_root
from omtra.load.conf import merge_task_spec
from omtra.dataset.data_module import MultiTaskDataModule
import hydra
import torch_cluster as tc
import dgl
from omtra.data.graph.utils import get_batch_idxs
from omtra.tasks.utils import get_edges_for_task
from omtra.data.graph import edge_types as all_edge_types
from omtra.data.graph import to_canonical_etype
import torch
from pathlib import Path

OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)

import warnings

# Suppress the specific warning from vlen_utf8.py
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    module="zarr.codecs.vlen_utf8"
)


# Absolute or relative path to your config directory
config_dir = Path(omtra_root()) / 'configs'
config_name = "config.yaml"  # or whatever your main config file is called

# Initialize Hydra and compose the config
# can odd overrides = [ list of string overrides]
overrides = [
"pharmit_path=/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit_dev",
"task_group=no_protein"
]
with initialize_config_dir(config_dir=os.path.abspath(config_dir)):
    cfg = compose(config_name=config_name, overrides=overrides)

cfg = merge_task_spec(cfg)

# Optional: print the config
# print(OmegaConf.to_yaml(cfg))

# load data module
datamodule: MultiTaskDataModule = hydra.utils.instantiate(
    cfg.task_group.datamodule, 
    # graph_config=cfg.graph,
    # prior_config=cfg.prior
)

train_dataset = datamodule.load_dataset("train")
pharmit_dataset = train_dataset.datasets['pharmit']



num_atom_types = len(lig_atom_type_map)
num_bond_orders = len(charge_map)

model = Model(num_atom_types=num_atom_types,    
                 num_atom_charges=num_bond_orders,
                 num_bond_orders=5, # TODO: no in constants.py
                 vector_size=4,
                 num_gvp_layers=1,
                 mlp_hidden_size=128,
                 embedding_dim=128,     
                 num_embeddings=100, 
                 num_decod_hiddens=256, 
                 num_decod_layers=3, 
                 num_bond_decod_hiddens=128, 
                 num_bond_decod_layers=3, 
                 commitment_cost=0.25)

""" 
Test 1: Passing one molecule through the model 
"""

idx = ('denovo_ligand', 0)
g = pharmit_dataset[idx]
display_graph(g)

model.eval()

with torch.no_grad():  # No gradient tracking is needed for inference
    loss, atom_type_logits, atom_charge_logits, bond_order_logits, perplexity = model(g)

print("Loss:", loss)
print("Perplexity:", perplexity)




"""
Test 2:  Train on a small batch
"""
class PharmitWrapperDataset(Dataset):
    def __init__(self, base_dataset, graph_type, length):
        self.base_dataset = base_dataset
        self.graph_type = graph_type
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.base_dataset[(self.graph_type, idx)]

wrapped_dataset = PharmitWrapperDataset(pharmit_dataset, 'denovo_ligand', length=10000)

def collate_fn(batch):
    return dgl.batch(batch)

batch_size = 100
training_loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


learning_rate = 1e-3    # From VQ-VAE paper

# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)


model.train()
train_res_error = []
train_res_perplexity = []

loader_iter = iter(training_loader)

for i in range(10):
    batched_graph = next(loader_iter)
    optimizer.zero_grad()

    loss, atom_types_hat, atom_charges_hat, bond_order_hat, perplexity = model(batched_graph)
    loss.backward()

    optimizer.step()

    train_res_error.append(loss.item())
    train_res_perplexity.append(perplexity.item())

    print('%d iterations' % (i+1))
    print(f"Loss: {np.mean(train_res_error[-2:]):.3f}")  # Show last 2 iterations
    print(f"Perplexity: {np.mean(train_res_perplexity[-2:]):.3f}")
    print()






