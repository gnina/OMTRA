import numpy as np
import dgl

import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from omtra.models.gvp import HeteroGVPConv, _rbf, _norm_no_nan
from omtra.constants import lig_atom_type_map, charge_map, bond_type_map, extra_feats_map
from omtra.data.graph.utils import get_upper_edge_mask
import pytorch_lightning as pl
import hydra
from pathlib import Path
from typing import Optional



class Encoder(nn.Module):
    def __init__(self, 
            a_embed_dim: int = 16,
            c_embed_dim: int = 8,
            e_embed_dim: int = 8,
            extra_feats_embed_dim: int = 4,
            scalar_size: int = 128, 
            vector_size: int = 4, 
            num_gvp_layers: int = 3, 
            latent_dim: int = 8, 
            rbf_dim: int = 10, 
            rbf_dmax: int = 32,
            mask_prob: float = 0.0,
            use_extra_feats: bool = False):
        super(Encoder, self).__init__()

        self.mask_prob = mask_prob
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.latent_dim = latent_dim
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.use_extra_feats = use_extra_feats


        self.a_embedding = nn.Embedding(num_embeddings=len(lig_atom_type_map)+int(mask_prob>0), embedding_dim=a_embed_dim)
        self.c_embedding = nn.Embedding(num_embeddings=len(charge_map)+int(mask_prob>0), embedding_dim=c_embed_dim)
        self.e_embedding = nn.Embedding(num_embeddings=len(bond_type_map), embedding_dim=e_embed_dim)
        
        
        if use_extra_feats:
            extra_feats_embedding = {}

            for feat, n_categories in extra_feats_map.items():
                extra_feats_embedding[feat] =  nn.Embedding(num_embeddings=n_categories+int(mask_prob>0), embedding_dim=extra_feats_embed_dim)
            
            self.extra_feats_embedding = nn.ModuleDict(extra_feats_embedding) 


            self.to_node_scalars = nn.Sequential(
                nn.Linear(a_embed_dim + c_embed_dim + len(extra_feats_map)*extra_feats_embed_dim, scalar_size),
                nn.ReLU(),
            )

        else:
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
        

    def forward(self, g, mask_prob=None):

        atom_types = g.nodes['lig'].data['a_1_true'].clone()    # Atom types, has shape (n_nodes,)
        atom_charges = g.nodes['lig'].data['c_1_true'].clone()  # Atom charges, has shape (n_nodes,)
        bond_orders = g.edges['lig_to_lig'].data['e_1_true']  # Bond orders, has shape (n_edges,)

        # apply random masking
        if mask_prob is None:
            mask_prob = self.mask_prob
        else:
            mask_prob = mask_prob

        mask = torch.rand_like(atom_types.float()) < mask_prob   # Binary mask
        atom_types[mask] = len(lig_atom_type_map)  # Set masked atom types to the last index (masked atom type)
        atom_charges[mask] = len(charge_map)  # Set masked atom charges to the last index (masked atom charge)

        # embed discrete data
        node_scalar_inputs = [self.a_embedding(atom_types), self.c_embedding(atom_charges)]  # (n_nodes, a_embed_dim + c_embed_dim)
            
        if self.use_extra_feats:
            for feat in extra_feats_map.keys():
                extra_atom_feat = g.nodes['lig'].data[feat+'_1_true'].clone()
                extra_atom_feat[mask] = extra_feats_map[feat]
                node_scalar_inputs.append(self.extra_feats_embedding[feat](extra_atom_feat))  # (n_nodes, a_embed_dim + c_embed_dim + n_extra_feats * extra_feats_embed_dim)
        
        scalar_feats = {
            'lig': self.to_node_scalars(torch.cat(node_scalar_inputs, dim=1))  # (n_nodes, scalar_size)
        }

        # embed edge data
        edge_feats = {
            'lig_to_lig': self.e_embedding(bond_orders)  # (n_edges, e_embed_dim)
        }


        ####
        # convert graph data into a format to be passed into the message-passing layers
        ####
        vector_feats = {'lig': torch.zeros((atom_types.shape[0], self.vector_size, 3), device=g.device)}    # Vector features
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
            scalar_feats, vector_feats = gvp_layer(g,
                               scalar_feats,
                               coord_feats,
                               vector_feats,
                               edge_feats,
                               x_diff,
                               d
                               )
            
        # convert ligand scalar features to latent atom types
        atom_latents = self.to_atom_latents(scalar_feats['lig'])  # (n_nodes, latent_dim)
            
        return atom_latents, mask



class VectorQuantizer(nn.Module):
    def __init__(self, 
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float):
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
        
        # Straight through estimator
        quantized = z_e + (quantized - z_e).detach()

        # Perplexity
        encodings = F.one_hot(encoding_indices, num_classes=self.embedding.num_embeddings).squeeze(1).float()
        avg_probs = torch.mean(encodings, dim=0)  
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))    #  "spread" of the quantized embeddings. Indicates how well the codebook is being used 
        
        # Number of unique codebook vectors used 
        unique_codes = torch.unique(encoding_indices)
        num_unique_codes = unique_codes.numel()

        return loss, quantized, encoding_indices, perplexity, num_unique_codes
    

class VectorQuantizerEMA(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int, 
                 commitment_cost: float,
                 decay: float = 0.99,
                 epsilon: float = 1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim) 
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))  # Running average of how frequently each codebook vector is used during training

        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))   
        self.ema_w.data.normal_()   # EMA of the summed inputs assigned to each codebook vector

        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon


    def forward(self, z_e):
        # z_e = (n_atoms, embedding_dim)
        # embedding = (num_embeddings, embedding_dim)

        distances = (torch.sum(z_e**2, dim=1, keepdim=True)         # (num_atoms, num_embeddings)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_e, self.embedding.weight.t()))
        
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # Get indices of closest codebook (embedding) vector
        encodings = F.one_hot(encoding_indices, num_classes=self.embedding.num_embeddings).squeeze(1).float()
        quantized = self.embedding(encoding_indices).squeeze(1)   # (num_atoms, embedding_dim)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), z_e)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))      

        # Loss
        commitment_loss = F.mse_loss(quantized.detach(), z_e)
        loss = self.commitment_cost * commitment_loss   # NOTE: no vector quantization loss when using EMA codebook updates
        
        # Straight through estimator
        quantized = z_e + (quantized - z_e).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)  
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))    #  "spread" of the quantized embeddings. Indicates how well the codebook is being used 
        
        # Number of unique codebook vectors used 
        unique_codes = torch.unique(encoding_indices)
        num_unique_codes = unique_codes.numel()

        return loss, quantized, encoding_indices, perplexity, num_unique_codes


class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 num_decod_hiddens: int, 
                 num_bond_decod_hiddens: int,
                 num_hidden_layers: int, 
                 rbf_dmax: int = 10,
                 rbf_dim: int = 32,
                 use_extra_feats: bool = False
                 ):
        super(Decoder, self).__init__()

        self.n_atom_types = len(lig_atom_type_map)
        self.n_atom_charges = len(charge_map)
        self.n_bond_orders = len(bond_type_map)

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.use_extra_feats = use_extra_feats


        # Atom type and charge decoder
        atom_layers = [nn.Linear(latent_dim, num_decod_hiddens),   # Initial layer
                       nn.ReLU()]    

        for _ in range(num_hidden_layers-1):    # Hidden layers
            atom_layers.append(nn.Linear(num_decod_hiddens, num_decod_hiddens))
            atom_layers.append(nn.ReLU())

        if use_extra_feats:
            atom_layers.append(nn.Linear(num_decod_hiddens, self.n_atom_types + self.n_atom_charges + sum(extra_feats_map.values())))   # Final layer
        else:
            atom_layers.append(nn.Linear(num_decod_hiddens, self.n_atom_types + self.n_atom_charges))   # Final layer
        
        self.atom_decoder = nn.Sequential(*atom_layers) # Use nn.Sequential to define the entire model
        
        # Bond order decoder
        bond_layers = [nn.Linear(latent_dim+rbf_dim, num_bond_decod_hiddens),   # Initial layer
                       nn.ReLU()]    

        for _ in range(num_hidden_layers-1):    # Hidden layers
            bond_layers.append(nn.Linear(num_bond_decod_hiddens, num_bond_decod_hiddens))
            bond_layers.append(nn.ReLU())

        bond_layers.append(nn.Linear(num_bond_decod_hiddens, self.n_bond_orders))   # Final layer
        self.bond_decoder = nn.Sequential(*bond_layers) # Use nn.Sequential to define the entire model
    

    def forward(self, g: dgl.DGLHeteroGraph, quantized: torch.Tensor):

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
        scalar_feats_logits = self.atom_decoder(quantized)  # (num_atoms, num_atom_types + num_atom_charges (+ num_extra_feats))
        logits = {}
        
        logit_sizes = {'atom_type': self.n_atom_types,
                        'atom_charge': self.n_atom_charges}
        
        if self.use_extra_feats:
            logit_sizes.update(extra_feats_map)
            
        start_idx = 0
        
        for feat, size in logit_sizes.items():
            end_idx = start_idx + size
            logits[feat] = scalar_feats_logits[:, start_idx:end_idx]
            start_idx = end_idx

        # predict bond orders for pairs of atoms
        pair_node_feats = quantized[pair_indices[:, 0]] + quantized[pair_indices[:, 1]]  # (P, latent_dim)
        combined_quantized_dists = torch.concat([pair_node_feats, dists], dim=1) # (P, rbf_dim + latent_dim)
        logits['bond_order'] = self.bond_decoder(combined_quantized_dists)    # (P, num_bond_orders)
       
        return logits
    


class LigandVQVAE(pl.LightningModule):
    def __init__(self,
                 scalar_size: int, # scalar features for message passing
                 vector_size: int, # vector features for message passing
                 num_gvp_layers: int,
                 latent_dim: int, # size of latent atom types
                 num_embeddings: int, # size of the codebook
                 num_decod_hiddens: int, 
                 num_bond_decod_hiddens: int,
                 num_hidden_layers: int, 
                 commitment_cost: int,
                 mask_prob: int,
                 a_embed_dim: int = 16,
                 c_embed_dim: int = 8,
                 e_embed_dim: int = 8,
                 extra_feats_embed_dim: int = 8,
                 rbf_dim: int = 32,
                 rbf_dmax: int = 10,                 
                 k_checkpoints: int = 20,
                 checkpoint_interval: int = 1000,
                 use_ema: bool =True,
                 decay: float = 0.99,
                 epsilon: float = 1e-5,
                 use_extra_feats: bool = False,
                 og_run_dir: Optional[str] = None,):
                 
        super().__init__()

        self.k_checkpoints: int = k_checkpoints
        self.checkpoint_interval: int = checkpoint_interval
        self.og_run_dir = og_run_dir

        self.n_mask_feats = int(mask_prob > 0)  # Number of masked features (atom types and charges)

        self.num_atom_types = len(lig_atom_type_map) 
        self.num_atom_charges = len(charge_map) 
        self.vector_size = vector_size
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.mask_prob = mask_prob
        self.use_extra_feats = use_extra_feats

        
        self.encoder = Encoder(
                               scalar_size=scalar_size, 
                               vector_size= vector_size,
                               a_embed_dim= a_embed_dim,
                               c_embed_dim= c_embed_dim,
                               e_embed_dim= e_embed_dim,
                               extra_feats_embed_dim= extra_feats_embed_dim,
                               num_gvp_layers= num_gvp_layers, 
                               latent_dim= latent_dim,
                               rbf_dim= rbf_dim,
                               rbf_dmax= rbf_dmax,
                               mask_prob= mask_prob,
                               use_extra_feats= use_extra_feats)


        if use_ema: # Exponential Moving Average codebook vector updates
            self.vq_vae = VectorQuantizerEMA(num_embeddings= num_embeddings,
                                             embedding_dim= latent_dim,
                                             commitment_cost= commitment_cost,
                                             decay= decay,
                                             epsilon= epsilon)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings= num_embeddings,
                                          embedding_dim= latent_dim,
                                          commitment_cost= commitment_cost)
            
        
        self.decoder = Decoder(latent_dim= latent_dim,
                               num_decod_hiddens= num_decod_hiddens,
                               num_bond_decod_hiddens= num_bond_decod_hiddens,
                               num_hidden_layers= num_hidden_layers,
                               rbf_dim= rbf_dim,
                               rbf_dmax= rbf_dmax,
                               use_extra_feats= use_extra_feats
                               )
        
        self.save_hyperparameters()
        self.configure_loss_fns()

    def manual_checkpoint(self, batch_idx: int):

        if batch_idx % self.checkpoint_interval == 0 and batch_idx != 0:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            log_dir = hydra_cfg['runtime']['output_dir']
            checkpoint_dir = Path(log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            current_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            current_checkpoints.sort(key=lambda x: x.stem.split("_")[-1])
            if len(current_checkpoints) >= self.k_checkpoints:
                # remove the oldest checkpoint
                oldest_checkpoint = current_checkpoints[0]
                oldest_checkpoint.unlink()

            checkpoint_path = checkpoint_dir / f'batch_{batch_idx}.ckpt'
            print('saving checkpoint to ', checkpoint_path, flush=True)
            self.trainer.save_checkpoint(str(checkpoint_path))
            print(f'Saved checkpoint to {checkpoint_path}')


    def configure_loss_fns(self):
        """ Define loss functions used in training """
        self.atom_type_loss_fn = torch.nn.CrossEntropyLoss()
        self.atom_charge_loss_fn = torch.nn.CrossEntropyLoss()

        self.bond_order_loss_fn = torch.nn.CrossEntropyLoss()

        if self.use_extra_feats:
            self.extra_feats_loss_fn = nn.ModuleDict({feat: torch.nn.CrossEntropyLoss() for feat in extra_feats_map.keys()})


    def forward(self, g: dgl.DGLHeteroGraph, mask_prob=None): 
        """ Get relevant features from batched graph """

        target_atom_types = g.nodes['lig'].data['a_1_true']  # (n_nodes,)
        target_atom_charges = g.nodes['lig'].data['c_1_true']  # (n_nodes,)
        upper_edge_mask = get_upper_edge_mask(g, 'lig_to_lig')  # (n_edges,)
        target_bond_orders = g.edges['lig_to_lig'].data['e_1_true'][upper_edge_mask]  # (n_edges,)   

        ####
        # Pass to VQ-VAE model
        ####
        z_e, _ = self.encoder(g, mask_prob=mask_prob)   # Encoding
        loss, z_d, _, perplexity, num_unique_codes = self.vq_vae(z_e)    # Vector Quantization
        logits = self.decoder(g, z_d)    # Decoding

        ####
        # Compute losses and accuracies
        ####
        atom_type_loss = self.atom_type_loss_fn(logits['atom_type'], target_atom_types)
        atom_charge_loss = self.atom_charge_loss_fn(logits['atom_charge'], target_atom_charges)
        bond_order_loss = self.bond_order_loss_fn(logits['bond_order'], target_bond_orders)

        # Accuracy
        a_accuracy = (torch.argmax(logits['atom_type'], dim=1) == target_atom_types).float().mean()
        c_accuracy = (torch.argmax(logits['atom_charge'], dim=1) == target_atom_charges).float().mean()
        e_accuracy = (torch.argmax(logits['bond_order'], dim=1) == target_bond_orders).float().mean()
        nonzero_indices = torch.nonzero(target_bond_orders != 0, as_tuple=True)[0]
        e_nonzero_accuracy = (torch.argmax(logits['bond_order'], dim=1)[nonzero_indices] == target_bond_orders[nonzero_indices]).float().mean()

        losses = {'vq+comittment_loss': loss,
                  'a_recon_loss': atom_type_loss,
                  'c_recon_loss': atom_charge_loss,
                  'e_recon_loss': bond_order_loss,
                  'a_accuracy': a_accuracy,
                  'c_accuracy': c_accuracy,
                  'e_accuracy': e_accuracy,
                  'e_non_zero_accuracy': e_nonzero_accuracy,
                  'perplexity': perplexity,
                  'num_unique_codes': num_unique_codes}

        if self.use_extra_feats:
            extra_feats_losses = {}

            for feat, loss_fn in self.extra_feats_loss_fn.items():
                # predicted classes
                target = g.nodes['lig'].data[feat+'_1_true']  # (n_nodes,)
                # loss
                extra_feats_losses[feat+'_recon_loss'] = loss_fn(logits[feat], target)
                # accuracy
                extra_feats_losses[feat+'_accuracy'] = (torch.argmax(logits[feat], dim=1) == target).float().mean()
            
            losses.update(extra_feats_losses)

        return losses
    

    def training_step(self, batch, batch_idx):
        g, task_name, dataset_name = batch

        self.manual_checkpoint(batch_idx)
        
        losses = self.forward(g)

        train_log_dict = {}
        for key, metric in losses.items():
            train_log_dict[f"{key}_train"] = metric

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for loss_name, loss_val in losses.items():
            if 'loss' in loss_name:
                total_loss = total_loss + 1.0 * loss_val

        self.log_dict(train_log_dict, sync_dist=True)
        self.log("train_total_loss", total_loss, prog_bar=True, sync_dist=True, on_step=True)

        return total_loss
    

    def validation_step(self, batch, batch_idx):
        g, task_name, dataset_name = batch

        self.manual_checkpoint(batch_idx)
    
        losses = self.forward(g, mask_prob=0.0)

        val_log_dict = {}
        for key, metric in losses.items():
            val_log_dict[f"{key}_val"] = metric

        total_loss = torch.zeros(1, device=g.device)
        for loss_name, loss_val in losses.items():
            if 'loss' in loss_name:
                total_loss = total_loss + 1.0 * loss_val

        self.log_dict(val_log_dict, sync_dist=True, batch_size=g.batch_size)
        self.log("val_total_loss", total_loss, prog_bar=True, sync_dist=True, on_step=True, batch_size=g.batch_size)

        return total_loss


    def configure_optimizers(self, lr=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

