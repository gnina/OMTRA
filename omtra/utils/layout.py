import torch
import dgl
from typing import Dict
from omtra.data.graph.utils import get_node_batch_idxs, get_edge_batch_idxs

from omtra.data.graph.utils import (
    get_upper_edge_mask,
)

class GraphLayout:
    """
    A class to compute and store layout information for converting between 
    DGL graphs and padded sequences.
    """
    
    def __init__(self, g: dgl.DGLHeteroGraph):
        """
        Initialize layout information from a DGL graph.
        
        Args:
            g: DGL heterogeneous graph
        """
        self.batch_size = g.batch_size
        self.device = g.device
        self.ntypes = g.ntypes
        
        # Compute layout information for each node type
        self.node_batch_idxs = {}
        self.node_offsets = {}
        self.max_nodes = {}
        self.num_nodes_per_graph = {}

        self.lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        self.node_batch_idxs = get_node_batch_idxs(g)
        self.edge_batch_idxs = get_edge_batch_idxs(g)
        
        for ntype in g.ntypes:
            if g.num_nodes(ntype) == 0:
                self.node_batch_idxs[ntype] = torch.empty(0, device=self.device, dtype=torch.long)
                self.node_offsets[ntype] = torch.empty(0, device=self.device, dtype=torch.long)
                self.max_nodes[ntype] = 0
                self.num_nodes_per_graph[ntype] = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
                continue
            
            # Get number of nodes per graph for this node type
            num_nodes_per_graph = g.batch_num_nodes(ntype)
            self.num_nodes_per_graph[ntype] = num_nodes_per_graph
            
            # Maximum nodes in any graph for this node type
            self.max_nodes[ntype] = num_nodes_per_graph.max().item()
            
            # Calculate node offsets for this node type
            node_offsets = torch.zeros_like(num_nodes_per_graph)
            node_offsets[1:] = num_nodes_per_graph[:-1].cumsum(dim=0)
            self.node_offsets[ntype] = node_offsets

    @classmethod
    def layout_and_pad(cls, g: dgl.DGLHeteroGraph, *args, **kwargs):
        layout = cls(g, *args, **kwargs)
        padded_node_feats, attention_masks = layout.graph_to_padded_sequence(g)
        return layout, padded_node_feats, attention_masks

    def graph_to_padded_sequence(
        self,
        g: dgl.DGLHeteroGraph,
        ):

        batch_size = g.batch_size
        device = g.device
        
        # Dictionary to store all padded features and attention masks
        padded_node_feats = {}
        attention_masks = {}
        
        # Process each node type
        for ntype in g.ntypes:
            if g.num_nodes(ntype) == 0:
                continue
                
            # Get number of nodes per graph for this node type
            num_nodes_per_graph = g.num_nodes(ntype)
            max_nodes = self.max_nodes[ntype]
            
            if max_nodes == 0:
                continue
                
            # Calculate node offsets for this node type
            node_offsets = self.node_offsets[ntype]
            
            # Get all node features for this node type
            node_data = g.nodes[ntype].data
            padded_node_feats[ntype] = {}

            n_nodes = g.num_nodes(ntype)
            node_ids = torch.arange(n_nodes, device=device)
            graph_ids = self.node_batch_idxs[ntype]
            node_pos_in_graph = node_ids - node_offsets[graph_ids]
            
            # Process each feature for this node type
            for feat_name, feat_tensor in node_data.items():
                feat_shape = feat_tensor.shape
                
                additional_dims = feat_shape[1:]  # All dimensions after the first
                
                # Create padded tensor with appropriate shape
                padded_shape = (batch_size, max_nodes) + additional_dims
                padded_feats = torch.zeros(padded_shape, device=device, dtype=feat_tensor.dtype)
                
                # Fill in the padded tensor
                padded_feats[graph_ids, node_pos_in_graph] = feat_tensor
                padded_node_feats[ntype][feat_name] = padded_feats
            
            # Create attention mask for this node type
            attention_mask = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
            attention_mask[graph_ids, node_pos_in_graph] = True

            attention_masks[ntype] = attention_mask
        
        return padded_node_feats, attention_masks

    def padded_sequence_to_graph(
        self,
        g: dgl.DGLHeteroGraph,
        padded_node_feats: Dict[str, Dict[str, torch.Tensor]],
        attention_masks: Dict[str, torch.Tensor] = None,
        inplace: bool = True
    ):
        """
        Convert padded node features back to DGL graph format.
        
        Args:
            g: DGL heterogeneous graph
            padded_node_feats: Dict mapping node types to dict of feature names to padded tensors
            attention_masks: Optional attention masks to validate which positions are valid
            inplace: If True, modify the graph in place. If False, return feature dict.
        
        Returns:
            If inplace=False, returns dict mapping node types to feature dicts
        """
        # Dictionary to store unpacked features (if not inplace)
        unpacked_features = {} if not inplace else None
        
        # Process each node type
        for ntype in padded_node_feats.keys():
            if g.num_nodes(ntype) == 0:
                continue
            
            n_nodes = g.num_nodes(ntype)
            node_ids = torch.arange(n_nodes, device=self.device)
            graph_ids = self.node_batch_idxs[ntype]
            node_pos_in_graph = node_ids - self.node_offsets[ntype][graph_ids]
            
            if not inplace:
                unpacked_features[ntype] = {}
            
            # Process each feature for this node type
            for feat_name, padded_tensor in padded_node_feats[ntype].items():
                # Extract the relevant features using advanced indexing
                unpacked_feats = padded_tensor[graph_ids, node_pos_in_graph]
                
                # Optional: validate against attention mask if provided
                # if attention_masks is not None and ntype in attention_masks:
                #     expected_mask = attention_masks[ntype][graph_ids, node_pos_in_graph]
                #     if not torch.all(expected_mask):
                #         print(f"Warning: Some features for {ntype}.{feat_name} are being extracted from masked positions")
                
                if inplace:
                    # Add features directly to the graph
                    g.nodes[ntype].data[feat_name] = unpacked_feats
                else:
                    # Store in return dictionary
                    unpacked_features[ntype][feat_name] = unpacked_feats
        
        if not inplace:
            return unpacked_features
        return g