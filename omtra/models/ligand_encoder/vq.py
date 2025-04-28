import numpy as np
import dgl

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from omtra.models.gvp import HeteroGVPConv, _rbf, _norm_no_nan
from omtra.constants import lig_atom_type_map, charge_map, bond_type_map
from omtra.data.graph.utils import get_upper_edge_mask
import pytorch_lightning as pl
    


class Encoder(nn.Module):
    def __init__(self, 
            a_embed_dim: int = 16,
            c_embed_dim: int = 8,
            e_embed_dim: int = 8,
            scalar_size: int = 128, 
            vector_size: int = 4, 
            num_gvp_layers: int = 3, 
            latent_dim: int = 8, 
            rbf_dim: int = 10, 
            rbf_dmax: int = 32,
            mask_prob: float = 0.0):
        super(Encoder, self).__init__()

        self.mask_prob = mask_prob
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.latent_dim = latent_dim
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax


        self.a_embedding = nn.Embedding(num_embeddings=len(lig_atom_type_map)+int(mask_prob>0), embedding_dim=a_embed_dim)
        self.c_embedding = nn.Embedding(num_embeddings=len(charge_map)+int(mask_prob>0), embedding_dim=c_embed_dim)
        self.e_embedding = nn.Embedding(num_embeddings=len(bond_type_map), embedding_dim=e_embed_dim)


        self.to_node_scalars = nn.Sequential(
            nn.Linear(a_embed_dim + c_embed_dim, scalar_size),
            nn.ReLU(),
        )
        
        edge_feat_size = {'lig_to_lig': e_embed_dim}
        self.gvps = nn.ModuleList([HeteroGVPConv(node_types=['lig'],
                                                edge_types=['lig_to_lig'],
                                                scalar_size=scalar_size,
                                                vector_size=vector_size,
                                                use_dst_feats= False,
                                                rbf_dim = rbf_dim,
                                                rbf_dmax= rbf_dmax,
                                                edge_feat_size=edge_feat_size,
                                                ) for _ in range(num_gvp_layers)])   
        
        self.to_atom_latents = nn.Sequential(nn.Linear(scalar_size, latent_dim*2),
                                 nn.ReLU(),
                                 nn.Linear(latent_dim*2, latent_dim))
        

    def forward(self, g):

        atom_types = g.nodes['lig'].data['a_1_true'].clone()    # Atom types, has shape (n_nodes,)
        atom_charges = g.nodes['lig'].data['c_1_true'].clone()  # Atom charges, has shape (n_nodes,)
        bond_orders = g.edges['lig_to_lig'].data['e_1_true']  # Bond orders, has shape (n_edges,)

        # apply random masking
        mask = torch.rand_like(atom_types.float()) < self.mask_prob   # Binary mask
        atom_types[mask] = len(lig_atom_type_map)  # Set masked atom types to the last index (masked atom type)
        atom_charges[mask] = len(charge_map)  # Set masked atom charges to the last index (masked atom charge)

        # embed discrete data
        node_scalar_inputs = [self.a_embedding(atom_types), self.c_embedding(atom_charges)]  # (n_nodes, a_embed_dim + c_embed_dim)
        scalar_feats = {
            'lig': self.to_node_scalars(torch.cat(node_scalar_inputs, dim=1))  # (n_nodes, scalar_size)
        }

        # embed edge data
        edge_feats = {
            'lig_to_lig': self.e_embedding(bond_orders)  # (n_edges, e_embed_dim)
        }



        ####
        # convert graph data into a format to be passed into the message-passing lyaers
        ####
        vector_feats = {'lig': torch.zeros((atom_types.shape[0], self.vector_size, 3))}    # Vector features
        coord_feats = {'lig': g.nodes['lig'].data['x_1_true']}   # Atom coordinates
        
        edges = g.edges(etype='lig_to_lig')
        diff = coord_feats['lig'][edges[0]] - coord_feats['lig'][edges[1]]
        d = _norm_no_nan(diff)  # (num_edges,)
        x_diff = diff / d.unsqueeze(1)   # (num_edges,)

        d_rbf = _rbf(d, D_min=0, D_max=self.rbf_dmax, D_count=self.rbf_dim) # _rbf:  D_min=0 and D_max=10, D_count=32
        d = {'lig_to_lig': d_rbf}   # Pairiwise distances 
        x_diff = {'lig_to_lig': x_diff} 

        # Sequentially pass through HeteroGVPConv layers
        for gvp_layer in self.gvps:
            scalar_feats, vector_feats = gvp_layer(g=g,
                               scalar_feats=scalar_feats,
                               coord_feats=coord_feats,
                               vec_feats= vector_feats,
                               edge_feats= edge_feats,
                               x_diff= x_diff,
                               d= d
                               )
            
        # convert ligand scalar features to latent atom types
        atom_latents = self.to_atom_latents(scalar_feats['lig'])  # (n_nodes, latent_dim)
            
        return atom_latents



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
    def __init__(self, 
                 latent_dim, 
                 num_decod_hiddens, 
                 num_bond_decod_hiddens, 
                 rbf_dmax=10,
                 rbf_dim=32,
                 ):
        super(Decoder, self).__init__()

        self.n_atom_types = len(lig_atom_type_map)
        self.n_atom_charges = len(charge_map)
        self.n_bond_orders = len(bond_type_map)
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        # Use nn.Sequential to define the entire model
        self.atom_decoder = nn.Sequential(
            nn.Linear(latent_dim, num_decod_hiddens),
            nn.ReLU(),
            nn.Linear(num_decod_hiddens, self.n_atom_types + self.n_atom_charges),
        )
        
        # Use nn.Sequential to define the entire model
        self.bond_decoder = nn.Sequential(
            nn.Linear(latent_dim+rbf_dim, num_bond_decod_hiddens),
            nn.ReLU(),
            nn.Linear(num_bond_decod_hiddens, self.n_bond_orders),
        )

    

    def forward(self, g:dgl.DGLHeteroGraph, quantized: torch.Tensor):

        lig_atom_positions = g.nodes['lig'].data['x_1_true'] # (N, d) Number of atoms

        # Get the number of 'lig' nodes in each graph component (this is a list or tensor)
        num_nodes_list = g.batch_num_nodes('lig')

        upper_edge_mask = get_upper_edge_mask(g, 'lig_to_lig')  # (num_edges,)

        src_idxs, dst_idxs = g.edges(etype='lig_to_lig')  # (2, num_edges)
        pair_indices = torch.stack([src_idxs, dst_idxs], dim=1)  # (num_edges, 2)
        pair_indices = pair_indices[upper_edge_mask]  # (num_pairs, 2)

        # Get the positions of each atom in the pair
        x = g.nodes['lig'].data['x_1_true']  # (N, 3)
        x_i = x[pair_indices[:, 0]]  # (P, 3)
        x_j = x[pair_indices[:, 1]]  # (P, 3)

        # Euclidean distance
        dists = _norm_no_nan(x_i - x_j)  # (P,)
        dists = _rbf(dists, D_min=0, D_max=self.rbf_dmax, D_count=self.rbf_dim)  # (P, rbf_dim)

        # project latent atom types to atom types and charges
        scalar_feats_logits = self.atom_decoder(quantized)  # (num_atoms, num_atom_types + num_atom_charges)
        atom_type_logits = scalar_feats_logits[:, :self.n_atom_types]    # (num_atoms, num_atom_types)
        atom_charge_logits = scalar_feats_logits[:, self.n_atom_types:]  # (num_atoms, num_atom_charges)

        # predict bond orders for pairs of atoms
        pair_node_feats = quantized[pair_indices[:, 0]] + quantized[pair_indices[:, 1]]  # (P, latent_dim)
        combined_quantized_dists = torch.concat([pair_node_feats, dists], dim=1) # (P, rbf_dim + latent_dim)
        bond_order_logits = self.bond_decoder(combined_quantized_dists)    # (P, num_bond_orders)
       
        return atom_type_logits, atom_charge_logits, bond_order_logits
    


class LigandVQVAE(pl.LightningModule):
    def __init__(self,
                 scalar_size: int, # scalar features for message passing
                 vector_size: int, # vector features for message passing
                 num_gvp_layers,
                 latent_dim, # size of latent atom types
                 num_embeddings, # size of the codebook
                 num_decod_hiddens, 
                 num_bond_decod_hiddens, 
                 commitment_cost,
                 a_embed_dim=16,
                 c_embed_dim=8,
                 e_embed_dim=8,
                 rbf_dim=32,
                 rbf_dmax=10,
                 mask_prob=0.10):
                 
        super().__init__()

        self.mask_prob = mask_prob
        self.n_mask_feats = int(mask_prob > 0)  # Number of masked features (atom types and charges)

        self.num_atom_types = len(lig_atom_type_map) 
        self.num_atom_charges = len(charge_map) 
        self.vector_size = vector_size
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.mask_prob = mask_prob
        
        self.encoder = Encoder(
                               scalar_size=scalar_size, 
                               vector_size = vector_size,
                               a_embed_dim= a_embed_dim,
                               c_embed_dim= c_embed_dim,
                               e_embed_dim= e_embed_dim,
                               num_gvp_layers= num_gvp_layers, 
                               latent_dim=latent_dim,
                               rbf_dim = self.rbf_dim,
                               rbf_dmax = self.rbf_dmax)

        self.vq_vae = VectorQuantizer(num_embeddings = num_embeddings,
                                      embedding_dim = latent_dim,
                                      commitment_cost = commitment_cost)
       
        self.decoder = Decoder(latent_dim = latent_dim,
                               num_decod_hiddens = num_decod_hiddens,
                               num_bond_decod_hiddens = num_bond_decod_hiddens,
                               rbf_dim=self.rbf_dim,
                               rbf_dmax=self.rbf_dmax
                               )
        
        self.save_hyperparameters()
        self.configure_loss_fns()


    def configure_loss_fns(self):
        """ Define loss functions used in training """
        self.atom_type_loss_fn = torch.nn.CrossEntropyLoss()
        self.atom_charge_loss_fn = torch.nn.CrossEntropyLoss()
        self.bond_order_loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, g: dgl.DGLHeteroGraph): 
        """ Get relevant features from batched graph """

        target_atom_types = g.nodes['lig'].data['a_1_true']  # (n_nodes,)
        target_atom_charges = g.nodes['lig'].data['c_1_true']  # (n_nodes,)
        upper_edge_mask = get_upper_edge_mask(g, 'lig_to_lig')  # (n_edges,)
        target_bond_orders = g.edges['lig_to_lig'].data['e_1_true'][upper_edge_mask]  # (n_edges,)


        """ Pass to VQ-VAE model """
        z_e = self.encoder(g)   # Encoding

        loss, z_d, perplexity = self.vq_vae(z_e)    # Vector Quantization
        
        atom_type_logits, atom_charge_logits, bond_order_logits = self.decoder(g, z_d)    # Decoding

        atom_type_loss = self.atom_type_loss_fn(atom_type_logits, target_atom_types)
        atom_charge_loss = self.atom_charge_loss_fn(atom_charge_logits, target_atom_charges)
        bond_order_loss = self.bond_order_loss_fn(bond_order_logits, target_bond_orders)

        # recon_loss = atom_type_loss + atom_charge_loss + bond_order_loss
        
        losses = {'vq+comittment': loss,
                  'a_recon': atom_type_loss,
                  'c_recon': atom_charge_loss,
                  'e_recon': bond_order_loss}

        return losses, atom_type_logits, atom_charge_logits, bond_order_logits, perplexity
    

    def training_step(self, batch, batch_idx):
        g = batch
        
        losses, atom_type_logits, atom_charge_logits, bond_order_logits, perplexity = self.forward(g)

        train_log_dict = {}
        for key, loss in losses.items():
            train_log_dict[f"{key}_train_loss"] = loss

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for loss_name, loss_val in losses.items():
            total_loss = total_loss + 1.0 * loss_val

        self.log_dict(train_log_dict, sync_dist=True)
        self.log("train_total_loss", total_loss, prog_bar=True, sync_dist=True, on_step=True)
        self.log("perplexity", perplexity, prog_bar=True, sync_dist=True, on_step=True)
        
        return total_loss
    

    def configure_optimizers(self, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer


def display_graph(g):
    """" Displays node and edge feature shapes for a dgl graph """

    print("\nNode Features:")
    print("x:", g.nodes['lig'].data['x_1_true'].shape)
    print("a:", g.nodes['lig'].data['a_1_true'].shape)
    print("c", g.nodes['lig'].data['c_1_true'].shape)
    print("e:", g.edges['lig_to_lig'].data['e_1_true'].shape)


""" Data Class """
if __name__ == "__main__":
    from omtra.load.quick import load_cfg, datamodule_from_config
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pharmit_path', type=str, default='/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit_dev')
    args = p.parse_args()

    overrides = [
    f"pharmit_path={args.pharmit_path}",
    "task_group=no_protein"
    ]
    cfg = load_cfg(overrides=overrides)
    datamodule = datamodule_from_config(cfg)

    train_dataset = datamodule.load_dataset("train")
    pharmit_dataset = train_dataset.datasets['pharmit']

    model = LigandVQVAE(
                    a_embed_dim=16,
                    c_embed_dim=8,
                    e_embed_dim=8,
                    scalar_size=128,  
                    vector_size=4,
                    num_gvp_layers= 2,
                    latent_dim=8,     
                    num_embeddings= 100, 
                    num_decod_hiddens=128, 
                    num_bond_decod_hiddens= 128, 
                    commitment_cost= 0.25)


    """ 
    Test 1: Passing one molecule through the model 
    """

    idx = ('denovo_ligand', 0)
    g = pharmit_dataset[idx]
    display_graph(g)

    model.eval()

    with torch.no_grad():  # No gradient tracking is needed for inference
        losses, atom_type_logits, atom_charge_logits, bond_order_logits, perplexity = model(g)

    print("Loss:", loss)
    print("Reconstruction Loss:", recon_loss)
    print("Perplexity:", perplexity, '\n')


    """
    Test 2:  Train on a small batch
    """
    class PharmitWrapperDataset(Dataset):
        def __init__(self, pharmit_dataset, graph_type):
            self.base_dataset = pharmit_dataset
            self.graph_type = graph_type
            self.length = len(pharmit_dataset)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return self.base_dataset[(self.graph_type, idx)]

    def collate_fn(batch):
        return dgl.batch(batch)

    # Wrapper so we can use PyTorch's DataLoader
    wrapped_dataset = PharmitWrapperDataset(pharmit_dataset, 'denovo_ligand')

    # Hyperparameters
    batch_size = 100
    learning_rate = 1e-3 

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train()
    train_res_error = []
    train_res_perplexity = []

    # Data Loader
    training_loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_iter = iter(training_loader)

    for i in range(10):
        batched_graph = next(loader_iter)
        optimizer.zero_grad()

        loss, recon_loss, atom_types_hat, atom_charges_hat, bond_order_hat, perplexity = model(batched_graph)
        loss.backward()

        optimizer.step()

        train_res_error.append(loss.item())
        train_res_perplexity.append(perplexity.item())

        print('%d iterations' % (i+1))
        print(f"Loss: {np.mean(train_res_error[-2:]):.3f}")  # Show last 2 iterations
        print(f"Perplexity: {np.mean(train_res_perplexity[-2:]):.3f}")
        print()






