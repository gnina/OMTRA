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
        layout = cls(g)
        padded_seqs_output = layout.graph_to_padded_sequence(g, *args, **kwargs)
        return layout, *padded_seqs_output

    def graph_to_padded_sequence(
        self,
        g: dgl.DGLHeteroGraph,
        allowed_feat_names=None
        ):

        batch_size = g.batch_size
        device = g.device
        
        # Dictionary to store all padded features and attention masks
        padded_node_feats = {}
        padded_edge_feats = {}
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

                if allowed_feat_names is not None and feat_name not in allowed_feat_names:
                    continue

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
        
        # Process each edge type
        for canonical_etype in g.canonical_etypes:
            src_type, etype, dst_type = canonical_etype
            
            if g.num_edges(etype) == 0:
                continue
            
            # Get max nodes for source and destination types
            max_src_nodes = self.max_nodes[src_type]
            max_dst_nodes = self.max_nodes[dst_type]
            
            if max_src_nodes == 0 or max_dst_nodes == 0:
                continue
            
            # Get edge data
            edge_data = g.edges[etype].data
            if len(edge_data) == 0:
                continue
            
            padded_edge_feats[etype] = {}
            
            # Get edge indices
            src_ids, dst_ids = g.edges(etype=etype)
            n_edges = len(src_ids)
            
            # Get batch indices for edges
            edge_batch_ids = self.edge_batch_idxs[etype]
            
            # Get positions of source and destination nodes in their padded sequences
            src_graph_ids = self.node_batch_idxs[src_type][src_ids]
            dst_graph_ids = self.node_batch_idxs[dst_type][dst_ids]
            
            src_offsets = self.node_offsets[src_type]
            dst_offsets = self.node_offsets[dst_type]
            
            src_pos_in_graph = src_ids - src_offsets[src_graph_ids]
            dst_pos_in_graph = dst_ids - dst_offsets[dst_graph_ids]
            
            # Process each edge feature
            for feat_name, feat_tensor in edge_data.items():

                if allowed_feat_names is not None and feat_name not in allowed_feat_names:
                    continue

                feat_shape = feat_tensor.shape
                additional_dims = feat_shape[1:]  # All dimensions after the first (edge dimension)
                
                # Create padded tensor with shape (B, N, M, D...)
                padded_shape = (batch_size, max_src_nodes, max_dst_nodes) + additional_dims
                padded_feats = torch.zeros(padded_shape, device=device, dtype=feat_tensor.dtype)
                
                # Fill in the padded tensor
                padded_feats[edge_batch_ids, src_pos_in_graph, dst_pos_in_graph] = feat_tensor
                padded_edge_feats[etype][feat_name] = padded_feats
        
        return padded_node_feats, attention_masks, padded_edge_feats

    def padded_sequence_to_graph(
        self,
        g: dgl.DGLHeteroGraph,
        padded_node_feats: Dict[str, Dict[str, torch.Tensor]] = None,
        attention_masks: Dict[str, torch.Tensor] = None,
        padded_edge_feats: Dict[str, Dict[str, torch.Tensor]] = None,
        inplace: bool = True,
        allowed_feat_names=None
    ):
        """
        Convert padded node and edge features back to DGL graph format.
        
        Args:
            g: DGL heterogeneous graph
            padded_node_feats: Dict mapping node types to dict of feature names to padded tensors
            attention_masks: Optional attention masks to validate which positions are valid
            padded_edge_feats: Dict mapping edge types to dict of feature names to padded tensors
            inplace: If True, modify the graph in place. If False, return feature dict.
            allowed_feat_names: Optional set of feature names to process. If None, process all.
        
        Returns:
            If inplace=False, returns tuple of dicts for unpacked node and edge features.
            If inplace=True, returns the modified graph.
        """
        # Dictionary to store unpacked features (if not inplace)
        unpacked_features = {} if not inplace else None
        
        # Process each node type
        if padded_node_feats is not None:
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

                    if allowed_feat_names is not None and feat_name not in allowed_feat_names:
                        continue

                    # Extract the relevant features using advanced indexing
                    unpacked_feats = padded_tensor[graph_ids, node_pos_in_graph]
                    
                    if inplace:
                        # Add features directly to the graph
                        g.nodes[ntype].data[feat_name] = unpacked_feats
                    else:
                        # Store in return dictionary
                        unpacked_features[ntype][feat_name] = unpacked_feats
        
        # Process each edge type
        unpacked_edge_features = {}
        if padded_edge_feats is not None:
            for etype in padded_edge_feats.keys():
                if g.num_edges(etype) == 0:
                    continue
                
                # Get edge indices
                src_ids, dst_ids = g.edges(etype=etype)
                n_edges = len(src_ids)
                
                # Get the source and destination types for this edge type
                if isinstance(etype, tuple):
                    src_type, _, dst_type = etype
                else:
                    # For homogeneous graphs or when etype is a string
                    canonical_etype = g.to_canonical_etype(etype)
                    src_type, _, dst_type = canonical_etype
                
                # Get batch indices and positions for edges
                edge_batch_ids = self.edge_batch_idxs[etype]
                
                src_graph_ids = self.node_batch_idxs[src_type][src_ids]
                dst_graph_ids = self.node_batch_idxs[dst_type][dst_ids]
                
                src_offsets = self.node_offsets[src_type]
                dst_offsets = self.node_offsets[dst_type]
                
                src_pos_in_graph = src_ids - src_offsets[src_graph_ids]
                dst_pos_in_graph = dst_ids - dst_offsets[dst_graph_ids]
                
                if not inplace and etype not in unpacked_features:
                    unpacked_edge_features[etype] = {}
                
                # Process each feature for this edge type
                for feat_name, padded_tensor in padded_edge_feats[etype].items():

                    if allowed_feat_names is not None and feat_name not in allowed_feat_names:
                        continue

                    # Extract the relevant features from (B, N, M, D...) to (K, D...)
                    unpacked_feats = padded_tensor[edge_batch_ids, src_pos_in_graph, dst_pos_in_graph]
                    
                    if inplace:
                        # Add features directly to the graph
                        g.edges[etype].data[feat_name] = unpacked_feats
                    else:
                        # Store in return dictionary
                        unpacked_edge_features[etype][feat_name] = unpacked_feats
        
        if not inplace:
            return unpacked_features, unpacked_edge_features
        return g
