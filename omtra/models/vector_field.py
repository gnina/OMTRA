import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch_scatter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import dgl
import dgl.function as fn
from collections import defaultdict
from typing import Union, Callable, Dict, Optional
from typing import List
from omegaconf import DictConfig
from omtra.models.gvp import HeteroGVPConv, GVP, _norm_no_nan, _rbf
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.models.self_conditioning import SelfConditioningResidualLayer
from omtra.tasks.tasks import Task
from omtra.tasks.utils import get_edges_for_task, build_edges, remove_edges
from omtra.tasks.register import task_name_to_class
from omtra.load.conf import TaskDatasetCoupling
from omtra.tasks.modalities import (
    Modality,
    name_to_modality,
)
from omtra.utils.ctmc import purity_sampling
from omtra.utils.embedding import get_time_embedding
from omtra.data.graph import to_canonical_etype, get_inv_edge_type
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    residue_map,
    protein_element_map,
    protein_atom_map,
)


from omtra.data.graph.utils import get_batch_idxs
from omtra.utils.graph import g_local_scope
from omtra.data.graph.layout import GraphLayout
from omtra.models.transformer import TransformerWrapper
from omtra.models.embeddings.pos_embed import get_pos_embedding

from omtra.priors.align import rigid_alignment

# from line_profiler import LineProfiler, profile

class VectorField(nn.Module):
    def __init__(
        self,
        interpolant_scheduler: InterpolantScheduler,
        td_coupling: TaskDatasetCoupling,
        graph_config: DictConfig,
        n_pharmvec_channels: int = 4,
        n_vec_channels: int = 16,
        n_cp_feats: int = 0,
        n_hidden_scalars: int = 64,
        n_hidden_edge_feats: int = 64,
        rbf_dmax=18,
        rbf_dim=32,
        time_embedding_dim: int = 64,
        token_dim: int = 64,
        dropout: float = 0.0,
        has_mask: bool = True,
        self_conditioning: bool = False,
        fake_atoms: bool = False,
        res_id_embed_dim: int = 64,
        n_heads=8,
        pos_emb: bool = False,
        n_pre_gvp_convs: int = 1,
    ):
        super().__init__()
        self.graph_config = graph_config
        self.token_dim = token_dim
        self.task_embedding_dim = token_dim
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.interpolant_scheduler = interpolant_scheduler
        self.td_coupling: TaskDatasetCoupling = td_coupling
        self.time_embedding_dim = time_embedding_dim
        self.self_conditioning = self_conditioning
        self.has_mask = has_mask
        self.fake_atoms = fake_atoms
        self.pos_emb = pos_emb

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        self.cat_temp_func = lambda t: 0.05

        assert n_vec_channels >= 3, "n_vec_channels must be >= 3"
        assert n_vec_channels >= 2 * n_pharmvec_channels, (
            "n_vec_channels must be >= 2*n_pharmvec_channels"
        )

        self.node_types = set()
        self.edge_types = set()
        self.token_embeddings = nn.ModuleDict()
        self.edge_feat_sizes = defaultdict(int)
        self.ntype_cat_feats = defaultdict(int)

        # get the set of all modalities that will be present in our graphs
        task_classes = [
            task_name_to_class(task_name) for task_name in td_coupling.task_space
        ]
        modality_present_space = set()
        modality_generated_space = set()
        for task_class in task_classes:
            for m in task_class.modalities_fixed:
                modality_present_space.add(m.name)
            for m in task_class.modalities_generated:
                modality_present_space.add(m.name)
                modality_generated_space.add(m.name)

        modality_present_space = sorted(list(modality_present_space))
        modality_generated_space = sorted(list(modality_generated_space))

        modalities_present_cls = [
            name_to_modality(modality_name) for modality_name in modality_present_space
        ]
        modalities_generated_cls = [
            name_to_modality(modality_name)
            for modality_name in modality_generated_space
        ]

        # get the set of all nodes present in our graphs
        self.node_types = set(
            m.entity_name for m in modalities_present_cls if m.graph_entity == "node"
        )
        self.node_types = sorted(list(self.node_types))

        # create token embeddings for all categorical features that we are modeling
        for m in modalities_present_cls:
            if not m.is_categorical:
                continue
            # if the modality is being generated, there is an extra mask token
            is_generated = m.name in modality_generated_space
            has_fake_atoms = (m.name == 'lig_a' or m.name == 'lig_cond_a') and self.fake_atoms 
            self.token_embeddings[m.name] = nn.Embedding(
                m.n_categories + int(is_generated) + int(has_fake_atoms), token_dim
            )
            # record the number of categorical features for each node type, not sure why, keeping tyler's code in place
            if m.graph_entity == "node":
                self.ntype_cat_feats[m.entity_name] += 1

        # record modalities that are on edges
        for m in modalities_present_cls:
            if m.graph_entity != "edge":
                continue
            if not m.is_categorical:
                raise ValueError("did not expect continuous edge features")
            self.edge_feat_sizes[m.entity_name] = n_hidden_edge_feats
            self.edge_types.add(m.entity_name)

        # get all edge types that we need to support
        # self.edge_types = set()
        for task in task_classes:
            self.edge_types.update(get_edges_for_task(task, graph_config))
            
        
        missing_inv_edges = set()
        for etype in self.edge_types:
            inv_etype = get_inv_edge_type(etype)
            if inv_etype not in self.edge_types:
                missing_inv_edges.add(inv_etype)
        if len(missing_inv_edges) > 0:
            print(f"missing inverse edges: {missing_inv_edges}")       
        # self.edge_types.update(missing_inv_edges)
        
        
        self.edge_types = sorted(list(self.edge_types))

        # create a task embedding
        self.task_embedding = nn.Embedding(
            len(td_coupling.task_space), self.task_embedding_dim
        )

        # for each node type, create a function for initial node embeddings
        self.scalar_embedding = nn.ModuleDict()
        for ntype in self.node_types:
            n_cat_feats = self.ntype_cat_feats[
                ntype
            ]  # number of categorical features for this node type
            input_dim = n_cat_feats * token_dim + self.time_embedding_dim + self.task_embedding_dim
            if res_id_embed_dim is not None and ntype == 'prot_atom':
                input_dim += res_id_embed_dim

            self.scalar_embedding[ntype] = nn.Sequential(
                nn.Linear(
                    input_dim,
                    n_hidden_scalars,
                ),
                nn.SiLU(),
                nn.LayerNorm(n_hidden_scalars),
            )

        # for each edge type that has edge features, create a function for initial edge embeddings
        self.edge_embedding = nn.ModuleDict()
        for etype in self.edge_types:
            if self.edge_feat_sizes[etype] == 0:
                continue
            self.edge_embedding[etype] = nn.Sequential(
                nn.Linear(token_dim, n_hidden_edge_feats),
                nn.SiLU(),
                nn.LayerNorm(n_hidden_edge_feats),
            )

        self.pre_convs = nn.ModuleList([])
        for _ in range(n_pre_gvp_convs):
            self.pre_convs.append(
                HeteroGVPConv(
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    scalar_size=n_hidden_scalars,
                    vector_size=n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    edge_feat_size=self.edge_feat_sizes,
                    n_message_gvps=3,
                    n_update_gvps=3,
                    message_norm=1,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    dropout=0.0,
                )
            )


        # edge embedding following pre-convs
        self.pre_edge_embedders = nn.ModuleDict()
        for etype in self.edge_types:
            if self.edge_feat_sizes[etype] == 0:
                continue
            self.pre_edge_embedders[etype] = EdgeUpdate(
                n_hidden_scalars,
                n_hidden_edge_feats,
                rbf_dim=rbf_dim,
            )

        self.transformer = TransformerWrapper(
                node_types=list(self.node_types),
                n_hidden_scalars=self.n_hidden_scalars,
                n_vec_channels=self.n_vec_channels,
                d_model=256, n_layers=4, n_heads=n_heads, 
                dim_ff=1024, use_residual=True,
                pair_dim=n_hidden_edge_feats,
                dropout=dropout,
            )

        # for every modality being generated that is a node position, create NodePositionUpdate layers
        self.node_position_updaters = nn.ModuleDict()
        for m in modalities_generated_cls:
            is_node_position = (
                m.graph_entity == "node" and m.data_key == "x"
            )
            if not is_node_position:
                continue
            ntype = m.entity_name
            self.node_position_updaters[ntype] = NodePositionUpdate(
                        n_hidden_scalars,
                        n_vec_channels,
                        n_gvps=3,
                        n_cp_feats=n_cp_feats,
                    )

        # for every edge modality being generated, create EdgeUpdate layers
        self.edge_updaters = nn.ModuleDict()
        for m in modalities_present_cls:
            if m.graph_entity != "edge":
                continue
            etype = m.entity_name
            if self.edge_feat_sizes[etype] == 0:
                # skip edges without edge features, although i don't think we shouuld
                # have edge features being generated that are empty
                continue
            self.edge_updaters[etype] = EdgeUpdate(
                        n_hidden_scalars,
                        n_hidden_edge_feats,
                        rbf_dim=rbf_dim,
                    )

        # need node output heads for node categorical features and node vector features.
        # node positions will be covered by the node update module.
        # TODO: node position updates interleaved with convs doesn't work well when some nodes are fixed and others are not
        # we could only support node position updates via a node_output_head...TBD
        self.node_output_heads = nn.ModuleDict()
        # loop over modalities on nodes that are being generated
        for m in modalities_generated_cls:
            is_node = m.graph_entity == "node"
            if not is_node:
                continue
            # if categorical, the output head is just a MLP on node scalar features
            if m.is_categorical:
                has_fake_atoms = (m.name == 'lig_a' or m.name == 'lig_cond_a') and self.fake_atoms # TODO: this breaks if using latent atom types + fake atoms
                self.node_output_heads[m.name] = nn.Sequential(
                    nn.Linear(n_hidden_scalars, n_hidden_scalars),
                    nn.SiLU(),
                    nn.Linear(n_hidden_scalars, m.n_categories+int(has_fake_atoms)),
                )
            elif m.data_key == "v":  # if a node vector feature
                # TODO: hard-coded assumption that this situation only applies to pharm
                # vector features, need to make this more general
                # also need to avoid hard-coding number of vec features out
                # also maybe this should be a 2-layer GVP instead of a 1-layer GVP
                self.node_output_heads[m.name] = GVP(
                    dim_feats_in=n_hidden_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_feats_out=4,
                    dim_vectors_out=4,
                )
            elif (
                m.data_key == "x"
            ):  # if a node position, we don't need to do anything to it
                continue
            else:
                raise ValueError("unaccounted for node feature type being generated")

        self.edge_output_heads = nn.ModuleDict()
        # need output head for edge types that we will predict bond order on
        for m in modalities_generated_cls:
            is_edge_feat = m.graph_entity == "edge"
            if not (is_edge_feat and m.is_categorical):
                continue
            self.edge_output_heads[m.name] = nn.Sequential(
                nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                nn.SiLU(),
                nn.Linear(n_hidden_edge_feats, m.n_categories),
            )

        if self.self_conditioning:
            # raise NotImplementedError("Self conditioning not implemented yet")
            self.self_conditioning_residual_layer = SelfConditioningResidualLayer(
                td_coupling=td_coupling,
                n_pharmvec_channels=n_pharmvec_channels,
                node_embedding_dim=n_hidden_scalars,
                edge_embedding_dim=n_hidden_edge_feats,
                rbf_dim=rbf_dim,
                rbf_dmax=rbf_dmax,
                res_id_embed_dim=res_id_embed_dim,
                fake_atoms=fake_atoms,
            )

    @g_local_scope
    def forward(
        self,
        g: dgl.DGLGraph,
        task_class: Task,
        t: torch.Tensor,
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
        apply_softmax=False,
        remove_com=False,
        prev_dst_dict=None,
        extract_latents_for_confidence=False,
    ):
        """Predict x_1 (trajectory destination) given x_t, and, optionally, previous destination features."""
        device = g.device

        node_scalar_features = {}
        node_positions = {}
        node_vec_features = {}
        edge_features = {}

        # the only edges that should be in g at this point are covalent, lig_to_lig, npnde_to_npnde, maybe pharm_to_pharm
        # need to add edges into protein structure, and between differing ntypes (except covalent)
        g = build_edges(g, task_class, node_batch_idx, self.graph_config)

        # compose the graph from modalities into the language of the GNN
        # that is node positions, node scalar features, node vector features, and edge features
        # our implicit assumption
        node_modalities = [m for m in task_class.modalities_present if m.is_node]
        edge_modalities = [
            m for m in task_class.modalities_present if not m.is_node
        ]

        # loop over modalities defined on nodes
        for modality in node_modalities:
            ntype = modality.entity_name
            if g.num_nodes(ntype) == 0:
                continue
            if modality.data_key == "x":  # set node positions
                node_positions[ntype] = g.nodes[ntype].data[
                    f"{modality.data_key}_t"
                ]
            if (
                modality.data_key == "v"
            ):  # set user-provided vector features as initial vector features
                vec_features = g.nodes[ntype].data[
                    f"{modality.data_key}_t"
                ]  # has shape (n_nodes, n_vecs_per_node, 3)
                n, v_in, _ = vec_features.shape
                assert self.n_vec_channels >= v_in
                # concatenate zeros to the end of the vector features until there are self.n_vec_channels channels
                vec_padding = torch.zeros(
                    n,
                    self.n_vec_channels - v_in,
                    3,
                    device=device,
                    dtype=vec_features.dtype,
                )
                vec_features = torch.cat([vec_features, vec_padding], dim=1)
                node_vec_features[ntype] = vec_features
            elif modality.is_categorical:  # node categorical features
                if ntype not in node_scalar_features:
                    node_scalar_features[ntype] = []
                node_scalar_features[ntype].append(
                    self.token_embeddings[modality.name](
                        g.nodes[ntype].data[f"{modality.data_key}_t"]
                    )
                )
            # should be positional encodings
            elif modality.data_key == "pos_enc":
                if ntype not in node_scalar_features:
                    node_scalar_features[ntype] = []
                node_scalar_features[ntype].append(
                    g.nodes[ntype].data[f"{modality.data_key}_t"]
                )

        # loop back over node types, and anything without vector features should be given a zero vector
        for ntype in node_scalar_features.keys():
            if ntype in node_vec_features:
                continue
            num_nodes = g.num_nodes(ntype)
            node_vec_features[ntype] = torch.zeros(
                (num_nodes, self.n_vec_channels, 3), device=device
            ).float()

        # now lets collect edge features
        for modality in edge_modalities:
            etype = modality.entity_name
            if self.edge_feat_sizes[etype] > 0 and g.num_edges(etype) > 0:
                edge_feats = self.token_embeddings[modality.name](
                    g.edges[etype].data[f"{modality.data_key}_t"]
                )
                edge_feats = self.edge_embedding[etype](edge_feats)
                edge_features[etype] = edge_feats

        # get task embedding
        task_idx = self.td_coupling.task_space.index(
            task_class.name
        )  # integer of the task index
        task_idx = torch.full(
            (g.batch_size,), task_idx, device=device
        )  # tensor of shape (batch_size) containing the task index
        task_embedding_batch = self.task_embedding(
            task_idx
        )  # tensor of shape (batch_size, token_dim)

        # add time and task embedding to node scalar features
        for ntype in node_scalar_features.keys():
            # add time embedding to node scalar features
            if self.time_embedding_dim == 1:
                node_scalar_features[ntype].append(
                    t[node_batch_idx[ntype]].unsqueeze(-1)
                )
            else:
                t_emb = get_time_embedding(t, embedding_dim=self.time_embedding_dim)
                t_emb = t_emb[node_batch_idx[ntype]]
                node_scalar_features[ntype].append(t_emb)

            node_scalar_features[ntype].append(
                task_embedding_batch[node_batch_idx[ntype]]
            )  # expand task embedding for each node in the batch

            # concatenate all initial node scalar features and pass through the embedding layer
            node_scalar_features[ntype] = torch.cat(
                node_scalar_features[ntype], dim=-1
            )
            node_scalar_features[ntype] = self.scalar_embedding[ntype](
                node_scalar_features[ntype]
            )

        if self.pos_emb:
            for ntype in node_scalar_features.keys():
                global_node_idx = torch.arange(
                    g.num_nodes(ntype), device=device
                )
                bnn = g.batch_num_nodes(ntype)
                rel_node_starts = torch.zeros(1+bnn.shape[0], device=bnn.device)
                rel_node_starts[1:] = torch.cumsum(bnn, dim=0)
                rel_node_starts = rel_node_starts[:-1]
                relative_node_idx = global_node_idx - rel_node_starts[node_batch_idx[ntype]]
                pos_emb = get_pos_embedding(
                    relative_node_idx,
                    self.n_hidden_scalars,
                )
                node_scalar_features[ntype] += pos_emb

        if self.self_conditioning and prev_dst_dict is None:
            train_self_condition = self.training and (torch.rand(1) > 0.5).item()
            inference_first_step = not self.training and (t == 0).all().item()

            # TODO: actually at the first inference step we can just not apply self conditioning, need to test performance effect
            if train_self_condition or inference_first_step:
                with torch.no_grad():
                    node_scalar_features_clone = {
                        ntype: feats.clone()
                        for ntype, feats in node_scalar_features.items()
                    }
                    node_vec_features_clone = {
                        ntype: feats.clone()
                        for ntype, feats in node_vec_features.items()
                    }
                    node_positions_clone = {
                        ntype: pos.clone() for ntype, pos in node_positions.items()
                    }
                    edge_features_clone = {
                        etype: feats.clone()
                        for etype, feats in edge_features.items()
                    }

                    prev_dst_dict = self.denoise_graph(
                        g,
                        task_class,
                        node_scalar_features_clone,
                        node_vec_features_clone,
                        node_positions_clone,
                        edge_features_clone,
                        node_batch_idx,
                        upper_edge_mask,
                        apply_softmax=True,
                        remove_com=False,
                    )

        if self.self_conditioning and prev_dst_dict is not None:
            (
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
            ) = self.self_conditioning_residual_layer(
                g,
                task_class,
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
                prev_dst_dict,
                node_batch_idx,
                upper_edge_mask,
            )

        dst_dict = self.denoise_graph(
            g,
            task_class,
            node_scalar_features,
            node_vec_features,
            node_positions,
            edge_features,
            node_batch_idx,
            upper_edge_mask,
            apply_softmax,
            remove_com,
            extract_latents_for_confidence=extract_latents_for_confidence,
        )

        # TODO: added this here for testing, but not sure if this is the right place
        g = remove_edges(g)
        
        if extract_latents_for_confidence:
            dst_dict, final_gnn_latents = dst_dict
            return dst_dict, final_gnn_latents
        else:
            return dst_dict

    # @profile
    def denoise_graph(
        self,
        g: dgl.DGLHeteroGraph,
        task_class: Task,
        node_scalar_features: Dict[str, torch.Tensor],
        node_vec_features: Dict[str, torch.Tensor],
        node_positions: Dict[str, torch.Tensor],
        edge_features: Dict[str, torch.Tensor],
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
        apply_softmax: bool = False,
        remove_com: bool = False,
        extract_latents_for_confidence=False,
    ):
        x_diff, d = self.precompute_distances(g)
        modalities_generated = task_class.modalities_generated

        # do some gvp convolutions before the transformer
        for conv in self.pre_convs:
            node_scalar_features, node_vec_features = conv(
                g,
                node_scalar_features,
                node_positions,
                node_vec_features,
                edge_features,
                d=d,
                x_diff=x_diff,
            )

        # update edge features after pre-convs
        for etype in edge_features:
            edge_features[etype] = self.pre_edge_embedders[etype](
                g,
                node_scalar_features,
                edge_features[etype],
                d=d[etype],
                etype=etype,
            )


        # pass through the transformer
        node_scalar_features, edge_features_out = self.transformer(
            g,
            scalar_feats=node_scalar_features,
            coord_feats=node_positions,
            vec_feats=node_vec_features,
            edge_feats=edge_features,
            x_diff=x_diff,
            d=d,
        )
        edge_features.update(edge_features_out)

        # iterate over positions being generated, update them
        ntypes_updated = set()
        for m in modalities_generated:
            is_position = m.graph_entity == "node" and m.data_key == "x"
            if not is_position:
                continue
            ntype = m.entity_name
            if g.num_nodes(ntype) == 0:
                continue
            node_positions[ntype] = self.node_position_updaters[ntype](
                node_scalar_features[ntype],
                node_positions[ntype],
                node_vec_features[ntype],
            )
            ntypes_updated.add(ntype)

        # recompute x_diff and d for the updated node positions
        for canonical_etype in g.canonical_etypes:
            if g.num_edges(canonical_etype) == 0:
                continue
            src_ntype, etype, dst_ntype = canonical_etype
            edges_need_update = src_ntype in ntypes_updated or dst_ntype in ntypes_updated
            if not edges_need_update:
                continue
            x_diff_etype, d_etype = self.precompute_distances(
                g, node_positions, etype=etype
            )
            x_diff.update(x_diff_etype)
            d.update(d_etype)
            

        # iterate over edge features being modeled and update them
        # implicit assumption here that edges with modalities defined on them are not being rebuilt
        for m in task_class.modalities_present:
            if m.is_node:
                continue
            etype = m.entity_name
            if g.num_edges(etype) == 0:
                continue
            edge_features[etype] = self.edge_updaters[etype](
                g,
                node_scalar_features,
                edge_features[etype],
                d=d[etype],
                etype=etype,
            )

        logits = {}
        for m in task_class.modalities_generated:
            if m.is_node and m.is_categorical:
                ntype = m.entity_name
                logits[m.name] = self.node_output_heads[m.name](
                    node_scalar_features[ntype]
                )
            elif not m.is_node and m.is_categorical:
                etype = m.entity_name
                ue_feats = edge_features[etype][upper_edge_mask[etype]]
                le_feats = edge_features[etype][~upper_edge_mask[etype]]
                logits[m.name] = self.edge_output_heads[m.name](
                    ue_feats + le_feats
                )

        # project node positions back into zero-COM subspace
        if remove_com:
            all_positions = []
            all_batch_idx = []

            for ntype in self.node_types:
                if g.num_nodes(ntype) == 0:
                    continue
                pos = node_positions[ntype]
                batch = node_batch_idx[ntype]
                all_positions.append(pos)
                all_batch_idx.append(batch)

            all_positions = torch.cat(all_positions, dim=0)
            all_batch_idx = torch.cat(all_batch_idx, dim=0)

            com = torch_scatter.scatter_mean(all_positions, all_batch_idx, dim=0)

            for ntype in self.node_types:
                if g.num_nodes(ntype) == 0:
                    continue
                batch = node_batch_idx[ntype]
                g.nodes[ntype].data["x_1_pred"] = node_positions[ntype]
                g.nodes[ntype].data["x_1_pred"] = (
                    g.nodes[ntype].data["x_1_pred"] - com[batch]
                )
                node_positions[ntype] = g.nodes[ntype].data["x_1_pred"]

        # build a dictionary of predicted features
        dst_dict = {}
        # modalities_present = task_class.modalities_fixed + task_class.modalities_generated
        for m in task_class.modalities_generated:
            if m.is_node and g.num_nodes(m.entity_name) == 0:
                continue
            if m.data_key == "x":
                dst_dict[m.name] = node_positions[m.entity_name]
            elif m.is_categorical:
                dst_dict[m.name] = logits[m.name]
                if apply_softmax:
                    dst_dict[m.name] = torch.softmax(
                        dst_dict[m.name], dim=-1
                    )
            elif m.data_key == "v":
                ntype = m.entity_name
                s_in = node_scalar_features[ntype]
                v_in = node_vec_features[ntype]
                _, v_out = self.node_output_heads[m.name]((s_in, v_in))
                dst_dict[m.name] = v_out
            else:
                raise NotImplementedError(f"unaccounted for modality: {m.name}")

        if extract_latents_for_confidence:
            pre_output_head_latents = {
                "node_scalar_features": node_scalar_features,
                "node_vec_features": node_vec_features, 
                "node_positions": node_positions,
                "edge_features": edge_features,
            }
            return dst_dict, pre_output_head_latents 
        else:
            return dst_dict

    @g_local_scope
    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None, etype=None):
        """Precompute the pairwise distances between all nodes in the graph."""
        x_diff = {}
        d = {}
        if not etype:
            etypes = self.edge_types
        else:
            if isinstance(etype, str):
                etypes = [etype]
            elif isinstance(etype, list):
                etypes = etype
            else:
                raise ValueError("etypes must be a string or a list of strings")
            
        for ntype in self.node_types:
            if g.num_nodes(ntype) == 0:
                continue
            if node_positions is None:
                g.nodes[ntype].data["x_d"] = g.nodes[ntype].data["x_t"]
            else:
                g.nodes[ntype].data["x_d"] = node_positions[ntype]

        for etype in etypes:
            if etype not in g.etypes or g.num_edges(etype) == 0:
                continue
            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"), etype=etype)
            dij = _norm_no_nan(g.edges[etype].data["x_diff"], keepdims=True) + 1e-8
            x_diff[etype] = g.edges[etype].data["x_diff"] / dij
            d[etype] = _rbf(
                dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim
            )

        return x_diff, d

    def integrate(
        self,
        g: dgl.DGLHeteroGraph,
        task: Task,
        upper_edge_mask: Dict[str, torch.Tensor],
        n_timesteps: int = 250,
        stochasticity: float = 30.0,
        cat_temp_func: Optional[Callable] = None,
        tspan=None,
        visualize=False,
        extract_latents_for_confidence=False,
        time_spacing: str = "even",
        **kwargs,
    ):
        # TODO: adapt flowmol integrate for hetero version
        # TODO: figure out what should be attribute of class vs passed as arg vs pulled from cfg etc, nail down defaults
    
        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func

        if tspan is None:
            if time_spacing == "even":
                t = torch.linspace(0, 1, n_timesteps, device=g.device)
            elif time_spacing == "uneven":
                # log-dense near t=1, then normalize to [0,1]
                t = 1.0 - torch.logspace(-2, 0, steps=n_timesteps + 1, device=g.device).flip(0)
                t = (t - t.min()) / (t.max() - t.min())  # vector-wise normalization
            else:
                raise ValueError(f"Unknown time_spacing: {time_spacing}")
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        # TODO: in FlowMol alpha_t and alpha_t_prime were just tensors, now they are dicts mapping modalities to the interpolant value
        # TODO: in FlowMol, we assumed there was an inteprolant alpha_t as the weight on the data distribution and that the weight on the prior was 1 - alpha_t
        # i want to make this more general, assume we have alpha_t (weight on data) and beta_t (weight on prior)...conditional path functions were already written
        # under this assumption, but i might have flipped alpha and beta from how i described them above
        alpha_t, beta_t = self.interpolant_scheduler.weights(t, task)
        alpha_t_prime, beta_t_prime = self.interpolant_scheduler.weight_derivative(t, task)

        if visualize:
            traj = defaultdict(list)
            def add_frame(g, traj=traj, task=task, first_frame=False):
                m_fixed = task.modalities_fixed
                for m in task.modalities_present:
                    if m.is_node:
                        if g.num_nodes(m.entity_name) == 0:
                            continue
                        data_src = g.nodes[m.entity_name]
                    else:
                        if g.num_edges(m.entity_name) == 0:
                            continue
                        data_src = g.edges[m.entity_name]
                    xt = data_src.data[f"{m.data_key}_t"]
                    
                    try:
                        xpred = data_src.data[f"{m.data_key}_1_pred"]
                    except KeyError:
                        if m in m_fixed and not first_frame:
                            # if this modality is fixed, we use its value at t as the "predicted value"
                            # this just ensures fixed modalities still "appear" in endpoint trajectories
                            xpred = data_src.data[f"{m.data_key}_t"]
                        else:
                            xpred = None

                    traj[m.name].append(xt.detach().clone().cpu())

                    if xpred is not None:
                        traj[f'{m.name}_pred'].append(xpred.detach().clone().cpu())

            add_frame(g, first_frame=True)

        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)

        dst_dict = None
        for s_idx in range(1, t.shape[0]):
            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = { k: alpha_t[k][s_idx - 1] for k in alpha_t }
            alpha_s_i = { k: alpha_t[k][s_idx] for k in alpha_t }
            alpha_t_prime_i = { k: alpha_t_prime[k][s_idx - 1] for k in alpha_t }
            beta_t_i = { k: beta_t[k][s_idx - 1] for k in beta_t }
            beta_s_i = { k: beta_t[k][s_idx] for k in beta_t }
            beta_t_prime_i = { k: beta_t_prime[k][s_idx - 1] for k in beta_t }

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False

            # compute next step and set x_t = x_s
            g, dst_dict = self.step(
                g=g,
                task=task,
                s_i=s_i,
                t_i=t_i,
                alpha_t_i=alpha_t_i,
                alpha_s_i=alpha_s_i,
                alpha_t_prime_i=alpha_t_prime_i,
                beta_t_i=beta_t_i,
                beta_s_i=beta_s_i,
                beta_t_prime_i=beta_t_prime_i,
                node_batch_idxs=node_batch_idxs,
                edge_batch_idxs=edge_batch_idxs,
                upper_edge_mask=upper_edge_mask,
                cat_temp_func=cat_temp_func,
                stochasticity=stochasticity,
                last_step=last_step,
                prev_dst_dict=dst_dict,
                extract_latents_for_confidence=extract_latents_for_confidence,
                **kwargs,
            )

            if visualize:
                add_frame(g)

        # set x_1 = x_t
        for modality in task.node_modalities_present:
            ntype = modality.entity_name
            dk = modality.data_key
            if g.num_nodes(ntype) == 0:
                continue
            g.nodes[ntype].data[f"{dk}_1"] = g.nodes[ntype].data[f"{dk}_t"]

        for modality in task.edge_modalities_present:
            etype = modality.entity_name
            dk = modality.data_key
            if g.num_edges(etype) == 0:
                continue
            g.edges[etype].data[f"{dk}_1"] = g.edges[etype].data[f"{dk}_t"]


        if not visualize:
            return g
        
        # if visualizing, generate trajectory dict for each example
        per_system_traj = [ {} for _ in range(g.batch_size) ]
        for m in task.modalities_present:
            if m.name not in traj or len(traj[m.name]) == 0:
                continue
            batch_traj = torch.stack(traj[m.name], dim=0) # tensor of shape (n_timesteps, n_nodes/n_edges, *)
            batch_pred_traj = torch.stack(traj[f'{m.name}_pred'], dim=0) 
            if m.is_node:
                split_locs = g.batch_num_nodes(ntype=m.entity_name).tolist()
            else:
                split_locs = g.batch_num_edges(etype=m.entity_name).tolist()
            per_graph_mtrajs = torch.split(batch_traj, split_locs, dim=1) # list of tensors of shape (n_timesteps, n_nodes/n_edges, *)
            per_graph_pred_mtrajs = torch.split(batch_pred_traj, split_locs, dim=1)
            for i in range(len(per_graph_mtrajs)):
                per_system_traj[i][m.name] = per_graph_mtrajs[i]
                per_system_traj[i][f'{m.name}_pred'] = per_graph_pred_mtrajs[i]
    
        return g, per_system_traj

    def step(
        self,
        g: dgl.DGLGraph,
        task: Task,
        s_i: torch.Tensor,
        t_i: torch.Tensor,
        alpha_t_i: Dict[str, torch.Tensor],
        alpha_s_i: Dict[str, torch.Tensor],
        alpha_t_prime_i: Dict[str, torch.Tensor],
        beta_t_i: Dict[str, torch.Tensor],
        beta_s_i: Dict[str, torch.Tensor],
        beta_t_prime_i: Dict[str, torch.Tensor],
        node_batch_idxs: Dict[str, torch.Tensor],
        edge_batch_idxs: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
        cat_temp_func: Callable,
        prev_dst_dict: Optional[Dict] = None,
        stochasticity: float = 8.0,
        last_step: bool = False,
        extract_latents_for_confidence=False,
        stochastic_sampling: bool = False,
        noise_scaler: float = 1.0,
        eps: float = 0.01,
    ):
        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        # predict the destination of the trajectory given the current timepoint
        vf_forward_output = self(
            g,
            task,
            t=torch.full((g.batch_size,), t_i, device=device),
            node_batch_idx=node_batch_idxs,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=False,  # TODO: is this ...should this be set to True?
            prev_dst_dict=prev_dst_dict,
            extract_latents_for_confidence=extract_latents_for_confidence
        )

        # TEMPORARY TEST: DOES ALIGNMENT ON EACH INFERENCE STEP HELP?
        # g.nodes['lig'].data['_xhat'] = vf_forward_output['lig_x']
        # g_unbatched = dgl.unbatch(g)
        # for g_i in g_unbatched:
        #     xt = g_i.nodes['lig'].data['x_t']
        #     xhat = g_i.nodes['lig'].data['_xhat']
        #     xhat_aligned = rigid_alignment(xhat, xt)
        #     g_i.nodes['lig'].data['_xhat'] = xhat_aligned
        # g = dgl.batch(g_unbatched)
        # vf_forward_output['lig_x'] = g.nodes['lig'].data['_xhat']
        # del g.nodes['lig'].data['_xhat']
        # END TEMPORARY TEST

        
        if extract_latents_for_confidence:
            dst_dict, model_latents = vf_forward_output
            # if the user requested latents, we rely on the plumbing we have in `forward` and `denoise_graph` to obtain model_latents, and populate it to the graph
            keys = ["node_scalar_features", "node_vec_features", "node_positions"] 
            
            # TODO: add "edge_features" if needed
            for key in keys:
                for ntype in model_latents[key]:
                    g.nodes[ntype].data[key] = model_latents[key][ntype]          
        else:
            dst_dict = vf_forward_output

        dt = s_i - t_i

        # get continuous and categorical modalities
        categorical_modalities = []
        continuous_modalities = []
        for m in task.modalities_generated:
            if m.is_categorical:
                categorical_modalities.append(m)
            else:
                continuous_modalities.append(m)

        # iterate over continuous modalities and apply updates
        for m in continuous_modalities:
            data_src = g.nodes if m.is_node else g.edges

            # skip if there are no nodes or edges of this type
            num_entries = (
                g.num_nodes(m.entity_name) if m.is_node else g.num_edges(m.entity_name)
            )
            if num_entries == 0:
                continue

            x_1 = dst_dict[m.name]
            x_t = data_src[m.entity_name].data[f"{m.data_key}_t"]
            vf = self.vector_field(x_t, x_1, alpha_t_i[m.name], alpha_t_prime_i[m.name], beta_t_i[m.name], beta_t_prime_i[m.name])

            if stochastic_sampling:
                g_t = 1 / (t_i + eps) if t_i < 0.90 else 0.0  # g(t_n-1)
                g_s = 1 / (s_i + eps) if s_i < 0.90 else 0.0  # g(t_n)
                score = (t_i * vf - x_t) / (1 - t_i)    # score(x_t_n-1, z)
                noise = torch.randn_like(x_t)

                data_src[m.entity_name].data[f"{m.data_key}_t"] = x_t + (vf + g_t*score)*dt + torch.sqrt(2 * dt * g_s * noise_scaler) * noise

            else:
                data_src[m.entity_name].data[f"{m.data_key}_t"] = x_t + dt * vf

            data_src[m.entity_name].data[f"{m.data_key}_1_pred"] = x_1.detach().clone()

        # iterate over categorical modalities and apply updates
        for m in categorical_modalities:
            if m.is_node:
                if g.num_nodes(m.entity_name) == 0:
                    continue
                data_src = g.nodes[m.entity_name].data
            else:
                if g.num_edges(m.entity_name) == 0:
                    continue
                data_src = g.edges[m.entity_name].data

            xt = data_src[f"{m.data_key}_t"]
            if not m.is_node:
                xt = xt[upper_edge_mask[m.entity_name]]

            p_s_1 = dst_dict[m.name]
            temperature = cat_temp_func(t_i)
            p_s_1 = nn.functional.softmax(
                torch.log(p_s_1) / temperature, dim=-1
            )  # log probabilities

            # TODO: other discrete sampling methods?
            # TODO: path planning, probably in place of purity sampling
            # TODO: campbell step assumes alpha_t = 1 - beta_t; need to change behavior if this is ever not the case
            has_fake_atoms = self.fake_atoms and m.name in ['lig_a', 'lig_cond_a']
            n_categories = m.n_categories + int(has_fake_atoms)

            xt, x_1_sampled = self.campbell_step(
                g=g,
                m=m,
                p_1_given_t=p_s_1,
                xt=xt,
                stochasticity=eta,
                beta_t=beta_t_i[m.name],
                beta_t_prime=beta_t_prime_i[m.name],
                dt=dt,
                batch_size=g.batch_size,
                batch_num_nodes=g.batch_num_edges(m.entity_name)
                if not m.is_node
                else g.batch_num_nodes(m.entity_name),
                mask_index=n_categories,
                last_step=last_step,
                # batch_idx=edge_batch_idxs[m.entity_name]
                # if not m.is_node
                # else node_batch_idxs[m.entity_name],
                upper_edge_mask=upper_edge_mask[m.entity_name]
                if not m.is_node
                else None,
            )

            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if not m.is_node:
                e_t = torch.zeros_like(data_src["e_t"])
                e_t[upper_edge_mask[m.entity_name]] = xt
                e_t[~upper_edge_mask[m.entity_name]] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(data_src["e_t"])
                e_1_sampled[upper_edge_mask[m.entity_name]] = x_1_sampled
                e_1_sampled[~upper_edge_mask[m.entity_name]] = x_1_sampled
                x_1_sampled = e_1_sampled

            data_src[f"{m.data_key}_t"] = xt
            data_src[f"{m.data_key}_1_pred"] = x_1_sampled

        return g, dst_dict
    
    def campbell_step(
        self,
        g: dgl.DGLHeteroGraph,
        m: Modality,
        p_1_given_t: torch.Tensor,
        xt: torch.Tensor,
        stochasticity: float,
        beta_t: torch.Tensor,
        beta_t_prime: torch.Tensor,
        dt,
        batch_size: int,
        batch_num_nodes: torch.Tensor,
        mask_index: int,
        last_step: bool,
        upper_edge_mask: Optional[torch.Tensor],
    ):
        x1 = Categorical(p_1_given_t).sample()  # has shape (num_nodes,)

        unmask_prob = dt * (beta_t_prime + stochasticity * beta_t) / (1 - beta_t)
        mask_prob = dt * stochasticity

        unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
        mask_prob = torch.clamp(mask_prob, min=0, max=1)

        # sample which nodes will be unmasked
        will_unmask = purity_sampling(
            g,
            m=m,
            xt=xt,
            x1_probs=p_1_given_t,
            unmask_prob=unmask_prob,
            mask_index=mask_index,
            batch_size=batch_size,
            batch_num_nodes=batch_num_nodes,
            device=xt.device,
            upper_edge_mask=upper_edge_mask,
        )

        # This is without purity sampling
        # # uniformly sample nodes to unmask
        # will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
        # will_unmask = will_unmask * (
        #     xt == mask_index
        # )  # only unmask nodes that are currently masked

        if not last_step:
            # compute which nodes will be masked
            will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
            will_mask = will_mask * (
                xt != mask_index
            )  # only mask nodes that are currently unmasked

            # mask the nodes
            xt[will_mask] = mask_index

        # unmask the nodes
        xt[will_unmask] = x1[will_unmask]

        return xt, x1

    def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime, beta_t, beta_t_prime):
        term_1 = alpha_t_prime/alpha_t*x_t
        term_2 = (alpha_t*beta_t_prime - beta_t*alpha_t_prime)/alpha_t*x_1
        vf = term_1 + term_2
        return vf

    def build_cat_temp_schedule(
        self, cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a
    ):
        if cat_temperature_schedule == "decay":
            cat_temp_func = lambda t: cat_temp_decay_max * torch.pow(
                1 - t, cat_temp_decay_a
            )
        elif isinstance(cat_temperature_schedule, (float, int)):
            cat_temp_func = lambda t: cat_temperature_schedule
        elif callable(cat_temperature_schedule):
            cat_temp_func = cat_temperature_schedule
        else:
            raise ValueError(
                f"Invalid cat_temperature_schedule: {cat_temperature_schedule}"
            )

        return cat_temp_func


class NodePositionUpdate(nn.Module):
    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        # TODO: should update_pos have > 1 layer?
        self.update_pos = nn.Linear(n_scalars, 3, bias=False)
        nn.init.zeros_(self.update_pos.weight)

        # self.gvps = []
        # for i in range(n_gvps):
        #     if i == n_gvps - 1:
        #         vectors_out = 1
        #         vectors_activation = nn.Identity()
        #     else:
        #         vectors_out = n_vec_channels
        #         vectors_activation = nn.Sigmoid()

        #     self.gvps.append(
        #         GVP(
        #             dim_feats_in=n_scalars,
        #             dim_feats_out=n_scalars,
        #             dim_vectors_in=n_vec_channels,
        #             dim_vectors_out=vectors_out,
        #             n_cp_feats=n_cp_feats,
        #             vectors_activation=vectors_activation,
        #         )
        #     )
        # self.gvps = nn.Sequential(*self.gvps)

    def forward(
        self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor
    ):
        vector_updates = self.update_pos(scalars)
        return positions + vector_updates
        # _, vector_updates = self.gvps((scalars, vectors))
        # return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):
    def __init__(
        self,
        n_node_scalars,
        n_edge_feats,
        rbf_dim=16,
    ):
        super().__init__()

        input_dim = n_node_scalars * 2 + n_edge_feats + rbf_dim

        self.edge_update_fn = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )
        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d, etype):
        src_ntype, _, dst_ntype = to_canonical_etype(etype)
        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges(etype=etype)

        mlp_inputs = [
            node_scalars[src_ntype][src_idxs],
            node_scalars[dst_ntype][dst_idxs],
            edge_feats,
            d
        ]

        edge_feats = self.edge_norm(
            edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1))
        )
        # edge_feats = edge_feats + self.edge_norm(self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats
