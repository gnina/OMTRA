import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from collections import defaultdict
from typing import Union, Callable, Dict, Optional
import scipy
from typing import List
from omtra.models.gvp import HeteroGVPConv, GVP, _norm_no_nan, _rbf
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import Modality, name_to_modality, MODALITY_ORDER
from omtra.utils.embedding import get_time_embedding
from omtra.utils.graph import canonical_node_features
from omtra.data.graph import to_canonical_etype
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    residue_map,
    protein_element_map,
    protein_atom_map,
)


class EndpointVectorField(nn.Module):
    def __init__(
        self,
        interpolant_scheduler: InterpolantScheduler,
        n_charges: int = 6,
        n_bond_types: int = 4,
        n_vec_channels: int = 16,
        n_cp_feats: int = 0,
        n_hidden_scalars: int = 64,
        n_hidden_edge_feats: int = 64,
        n_recycles: int = 1,
        n_molecule_updates: int = 2,
        convs_per_update: int = 2,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        n_expansion_gvps: int = 3,
        separate_mol_updaters: bool = False,
        message_norm: Union[float, str] = 100,
        update_edge_w_distance: bool = False,
        rbf_dmax=20,
        rbf_dim=16,
        continuous_inv_temp_schedule=None,
        continuous_inv_temp_max: float = 10.0,
        time_embedding_dim: int = 64,
        token_dim: int = 64,
        attention: bool = False,
        n_heads: int = 1,
        s_message_dim: int = None,
        v_message_dim: int = None,
        dropout: float = 0.0,
        has_mask: bool = True,
        self_conditioning: bool = False,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4,
        # if we are using CTMC, input categorical features will have mask tokens,
        # this means their one-hot representations will have an extra dimension,
        # and the neural network instantiated by this method need to account for this
        # it is definitely anti-pattern to have a parameter in parent class that is only needed for one sub-class (CTMCVectorField)
        # however, this is the fastest way to get CTMCVectorField working right now, so we will be anti-pattern for the sake of time
    ):
        super().__init__()
        self.token_dim = token_dim
        self.n_lig_atom_types = len(lig_atom_type_map)
        self.n_npnde_atom_types = len(npnde_atom_type_map)
        self.n_protein_atom_types = len(protein_atom_map)
        self.n_protein_residue_types = len(residue_map)
        self.n_protein_element_types = len(protein_element_map)
        self.n_pharm_types = len(ph_idx_to_type)
        self.n_cross_edge_types = (
            2  # NOTE: un-hard code eventually (2 for proximity, covalent)
        )
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.interpolant_scheduler = interpolant_scheduler
        self.time_embedding_dim = time_embedding_dim
        self.self_conditioning = self_conditioning
        self.has_mask = has_mask

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, "n_vec_channels must be >= 3"

        self.continuous_inv_temp_schedule = continuous_inv_temp_schedule
        self.continouts_inv_temp_max = continuous_inv_temp_max
        self.continuous_inv_temp_func = self.build_continuous_inv_temp_func(
            self.continuous_inv_temp_schedule, self.continouts_inv_temp_max
        )

        self.n_cat_feats = {  # number of possible values for each categorical variable (+1 for mask token for generated modalities)
            "lig_a": self.n_lig_atom_types + 1,
            "lig_c": self.n_charges + 1,
            "lig_to_lig": self.n_bond_types + 1,
            "lig_e": self.n_bond_types + 1,
            "npnde_a": self.n_npnde_atom_types,
            "npnde_c": self.n_charges,
            "npnde_to_npnde": self.n_bond_types,
            "npnde_e": self.n_bond_types,
            "pharm_a": self.n_pharm_types + 1,
            "prot_atom_name": self.n_protein_atom_types,
            "prot_atom_element": self.n_protein_element_types,
            "prot_atom_r": self.n_protein_residue_types,
            "prot_res_r": self.n_protein_residue_types,
            "prot_atom_to_lig": self.n_cross_edge_types,
            "prot_atom_to_npnde": self.n_cross_edge_types,
            "prot_res_to_lig": self.n_cross_edge_types,
            "prot_res_to_npnde": self.n_cross_edge_types,
        }
        self.node_types = set()
        self.edge_types = set()
        self.token_embeddings = nn.ModuleDict()
        self.edge_feat_sizes = defaultdict(int)
        self.ntype_cat_feats = defaultdict(int)

        for modality_name in MODALITY_ORDER:
            modality = name_to_modality(modality_name)
            if modality.data_key == "x" or modality.data_key == "v":
                continue
            self.token_embeddings[modality_name] = nn.Embedding(
                self.n_cat_feats[modality_name], token_dim
            )
            if modality.graph_entity == "edge":
                self.edge_feat_sizes[modality.entity_name] = n_hidden_edge_feats
                self.edge_types.add(modality.entity_name)
            else:
                self.node_types.add(modality.entity_name)
                self.ntype_cat_feats[modality.entity_name] += 1

        self.scalar_embedding = nn.ModuleDict()
        self.edge_embedding = nn.ModuleDict()

        for ntype in self.node_types:
            i = self.ntype_cat_feats[ntype]
            self.scalar_embedding[ntype] = nn.Sequential(
                nn.Linear(
                    i * token_dim + self.time_embedding_dim,
                    n_hidden_scalars,
                ),
                nn.SiLU(),
                nn.Linear(n_hidden_scalars, n_hidden_scalars),
                nn.SiLU(),
                nn.LayerNorm(n_hidden_scalars),
            )

        for etype in self.edge_types:
            if self.edge_feat_sizes[etype] > 0:
                self.edge_embedding[etype] = nn.Sequential(
                    nn.Linear(token_dim, n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                    nn.SiLU(),
                    nn.LayerNorm(n_hidden_edge_feats),
                )

        self.conv_layers = []
        for conv_idx in range(convs_per_update * n_molecule_updates):
            self.conv_layers.append(
                HeteroGVPConv(
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    scalar_size=n_hidden_scalars,
                    vector_size=n_vec_channels,
                    n_cp_feats=n_cp_feats,
                    edge_feat_size=self.edge_feat_sizes,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    n_expansion_gvps=n_expansion_gvps,
                    message_norm=message_norm,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    attention=attention,
                    n_heads=n_heads,
                    s_message_dim=s_message_dim,
                    v_message_dim=v_message_dim,
                    dropout=dropout,
                    use_dst_feats=use_dst_feats,
                    dst_feat_msg_reduction_factor=dst_feat_msg_reduction_factor,
                )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleDict()
        self.edge_updaters = nn.ModuleDict()
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for ntype in self.node_types:
            self.node_position_updaters[ntype] = nn.ModuleList()
            for _ in range(n_updaters):
                self.node_position_updaters[ntype].append(
                    NodePositionUpdate(
                        n_hidden_scalars,
                        n_vec_channels,
                        n_gvps=3,
                        n_cp_feats=n_cp_feats,
                    )
                )

        for etype in self.edge_types:
            if self.edge_feat_sizes[etype] > 0:
                self.edge_updaters[etype] = nn.ModuleList()
                for _ in range(n_updaters):
                    self.edge_updaters[etype].append(
                        EdgeUpdate(
                            n_hidden_scalars,
                            n_hidden_edge_feats,
                            update_edge_w_distance=update_edge_w_distance,
                            rbf_dim=rbf_dim,
                        )
                    )

        self.node_output_heads = (
            nn.ModuleDict()
        )  # only need node output heads for cat modalities generated
        for ntype in ["lig", "pharm"]:
            output_dim = 0
            if ntype == "lig":
                output_dim = self.n_lig_atom_types + n_charges
            elif ntype == "pharm":
                output_dim = self.n_pharm_types
            self.node_output_heads[ntype] = nn.Sequential(
                nn.Linear(n_hidden_scalars, n_hidden_scalars),
                nn.SiLU(),
                nn.Linear(n_hidden_scalars, output_dim),
            )

        self.edge_output_heads = (
            nn.ModuleDict()
        )  # need output head for edge types that we will predict bond order on
        for etype in [
            "lig_to_lig",
            "prot_atom_to_lig",
            "prot_atom_to_npnde",
            "prot_res_to_lig",
            "prot_res_to_npnde",
        ]:
            output_dim = self.n_cross_edge_types
            if etype == "lig_to_lig":
                output_dim = self.n_bond_types
            self.edge_output_heads[etype] = nn.Sequential(
                nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
                nn.SiLU(),
                nn.Linear(n_hidden_edge_feats, output_dim),
            )

        if self.self_conditioning:
            raise NotImplementedError("Self conditioning not implemented yet")
            self.self_conditioning_residual_layer = SelfConditioningResidualLayer(
                n_atom_types=n_atom_types,
                n_charges=n_charges,
                n_bond_types=n_bond_types,
                node_embedding_dim=n_hidden_scalars,
                edge_embedding_dim=n_hidden_edge_feats,
                rbf_dim=rbf_dim,
                rbf_dmax=rbf_dmax,
            )

    def build_continuous_inv_temp_func(self, schedule, max_inv_temp=None):
        if schedule is None:
            inv_temp_func = lambda t: 1.0
        elif schedule == "linear":
            inv_temp_func = lambda t: max_inv_temp * (1 - t)
        elif callable(schedule):
            inv_temp_func = schedule
        else:
            raise ValueError(f"Invalid continuous_inv_temp_schedule: {schedule}")
        return inv_temp_func

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
    ):
        """Predict x_1 (trajectory destination) given x_t, and, optionally, previous destination features."""
        device = g.device

        with g.local_scope():
            node_scalar_features = {}
            node_positions = {}
            node_vec_features = {}
            edge_features = {}

            modalities_present = (
                task_class.modalities_fixed + task_class.modalities_generated
            )
            for modality in modalities_present:
                if modality.graph_entity == "node":
                    ntype = modality.entity_name
                    if modality.data_key == "x":
                        node_positions[ntype] = g.nodes[ntype].data[
                            f"{modality.data_key}_t"
                        ]
                        num_nodes = g.num_nodes(ntype)
                        node_vec_features[ntype] = torch.zeros(
                            (num_nodes, self.n_vec_channels, 3), device=device
                        )
                    else:
                        if ntype not in node_scalar_features:
                            node_scalar_features[ntype] = []
                        node_scalar_features[ntype].append(
                            self.token_embeddings[modality.name](
                                g.nodes[ntype]
                                .data[f"{modality.data_key}_t"]
                                .argmax(
                                    dim=-1
                                )  # NOTE: this assumes that the input is one-hot encoded
                            )
                        )
                else:
                    etype = modality.entity_name
                    if self.edge_feat_sizes[etype] > 0:
                        edge_feats = self.token_embeddings[modality.entity_name](
                            g.edges[etype]
                            .data[f"{modality.data_key}_t"]
                            .argmax(
                                dim=-1
                            )  # NOTE: this assumes that the input is one-hot encoded
                        )
                        edge_feats = self.edge_embedding[etype](edge_feats)
                        edge_features[etype] = edge_feats

            for ntype in node_scalar_features.keys():
                if self.time_embedding_dim == 1:
                    node_scalar_features[ntype].append(
                        t[node_batch_idx[ntype]].unsqueeze(-1)
                    )
                else:
                    t_emb = get_time_embedding(t, embedding_dim=self.time_embedding_dim)
                    t_emb = t_emb[node_batch_idx[ntype]]
                    node_scalar_features[ntype].append(t_emb)

                node_scalar_features[ntype] = torch.cat(
                    node_scalar_features[ntype], dim=-1
                )
                node_scalar_features[ntype] = self.scalar_embedding[ntype](
                    node_scalar_features[ntype]
                )

            if self.self_conditioning and prev_dst_dict is None:
                train_self_condition = self.training and (torch.rand(1) > 0.5).item()
                inference_first_step = not self.training and (t == 0).all().item()

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
                # TODO: Adapt self-conditioning residual layer for Hetero graph
                (
                    node_scalar_features,
                    node_positions,
                    node_vec_features,
                    edge_features,
                ) = self.self_conditioning_residual_layer(
                    g,
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
            )

            return dst_dict

    def denoise_graph(
        self,
        g: dgl.DGLGraph,
        task_class: Task,
        node_scalar_features: Dict[str, torch.Tensor],
        node_vec_features: Dict[str, torch.Tensor],
        node_positions: Dict[str, torch.Tensor],
        edge_features: Dict[str, torch.Tensor],
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
        apply_softmax: bool = False,
        remove_com: bool = False,
    ):
        x_diff, d = self.precompute_distances(g)
        for recycle_idx in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                # perform a single convolution which updates node scalar and vector features (but not positions)
                node_scalar_features, node_vec_features = conv(
                    g,
                    scalar_feats=node_scalar_features,
                    coord_feats=node_positions,
                    vec_feats=node_vec_features,
                    edge_feats=edge_features,
                    x_diff=x_diff,
                    d=d,
                )
                # every convs_per_update convolutions, update the node positions and edge features
                if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                    if self.separate_mol_updaters:
                        updater_idx = conv_idx // self.convs_per_update
                    else:
                        updater_idx = 0

                    modalities_generated = task_class.modalities_generated
                    for modality in modalities_generated:
                        if modality.graph_entity == "node" and modality.data_key == "x":
                            ntype = modality.entity_name
                            node_positions[ntype] = self.node_position_updaters[ntype][
                                updater_idx
                            ](
                                node_scalar_features[ntype],
                                node_positions[ntype],
                                node_vec_features[ntype],
                            )
                        if modality.graph_entity == "edge":
                            x_diff, d = self.precompute_distances(
                                g, node_positions
                            )  # NOTE: consider adding etype arg to precompute dists
                            etype = modality.entity_name
                            edge_features[etype] = self.edge_updaters[etype][
                                updater_idx
                            ](
                                g,
                                node_scalar_features,
                                edge_features[etype],
                                d=d[etype],
                                etype=etype,
                            )

        # predict final charges and atom type logits
        logits = {}
        for ntype in [
            "lig",
            "pharm",
        ]:  # TODO: eventually consider reading this from Task groups_generated/modalities
            node_scalar_features[ntype] = self.node_output_heads[ntype](
                node_scalar_features[ntype]
            )

            if ntype == "lig":
                logits["lig_a"] = node_scalar_features[ntype][:, : self.n_atom_types]
                if not self.exclude_charges:
                    logits["lig_c"] = node_scalar_features[ntype][
                        :, self.n_atom_types :
                    ]
            else:
                logits["pharm_a"] = node_scalar_features[ntype]

        # predict the final edge logits
        edge_logits = {}
        for etype in self.edge_types:
            ue_feats = edge_features[etype][upper_edge_mask[etype]]
            le_feats = edge_features[etype][~upper_edge_mask[etype]]
            edge_logits[etype] = self.edge_output_heads[etype](ue_feats + le_feats)

        # project node positions back into zero-COM subspace
        if remove_com:
            for ntype in self.node_types:
                g.nodes[ntype].data["x_1_pred"] = node_positions[ntype]
                g.nodes[ntype].data["x_1_pred"] = (
                    g.nodes[ntype].data["x_1_pred"]
                    - dgl.readout_nodes(g, feat="x_1_pred", op="mean", ntype=ntype)[
                        node_batch_idx[ntype]
                    ]
                )
                node_positions[ntype] = g.nodes[ntype].data["x_1_pred"]

        # build a dictionary of predicted features
        dst_dict = {}
        dst_dict["nodes"] = {}
        for ntype in ["lig", "pharm"]:
            dst_dict["nodes"][ntype] = {}
            for feat in canonical_node_features[ntype]:
                if feat == "x":
                    dst_dict["nodes"][ntype][feat] = node_positions[ntype]
                else:
                    if f"{ntype}_{feat}" in logits:
                        dst_dict["nodes"][ntype][feat] = logits[f"{ntype}_{feat}"]
        dst_dict["edges"] = {}
        for etype in self.edge_types:
            if etype in edge_logits:
                dst_dict["edges"][etype] = edge_logits[etype]

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for ntype, featdict in dst_dict["nodes"].items():
                if feat in ["a", "c"]:  # if this is a categorical feature
                    dst_dict["nodes"][ntype][feat] = torch.softmax(
                        dst_dict["nodes"][ntype][feat], dim=-1
                    )  # apply softmax to this feature
            for etype, logits in dst_dict["edges"].items():
                dst_dict["edges"][etype] = torch.softmax(logits, dim=-1)

        return dst_dict

    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""
        x_diff = {}
        d = {}
        with g.local_scope():
            for ntype in self.node_types:
                if node_positions is None:
                    g.nodes[ntype].data["x_d"] = g.nodes[ntype].data["x_t"]
                else:
                    g.nodes[ntype].data["x_d"] = node_positions[ntype]

            for etype in self.edge_types:
                g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"), etype=etype)
                dij = _norm_no_nan(g.edges[etype].data["x_diff"], keepdims=True) + 1e-8
                x_diff[etype] = g.edges[etype].data["x_diff"] / dij
                d[etype] = _rbf(
                    dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim
                )

        return x_diff, d

    def integrate(
        self,
        g: dgl.DGLGraph,
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
        n_timesteps: int,
        visualize=False,
        **kwargs,
    ):
        # TODO: adapt flowmol integrate for hetero version
        t = torch.linspace(0, 1, n_timesteps, device=g.device)

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(
            t
        )  # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        for ntype, featlist in canonical_node_features.items():
            for feat in featlist:
                g.nodes[ntype].data[f"{feat}_t"] = g.nodes[ntype].data[f"{feat}_0"]

        for etype in self.edge_types:
            g.edges[etype].data["e_t"] = g.edges[etype].data["e_0"]

        if visualize:
            traj_frames = {}
            for ntype in self.node_types:
                split_sizes = g.batch_num_nodes(ntype).detach().cpu().tolist()

                for feat in canonical_node_features[ntype]:
                    feat_key = f"{ntype}_{feat}"
                    init_frame = g.nodes[ntype].data[f"{feat}_0"].detach().cpu()
                    init_frame = torch.split(init_frame, split_sizes)
                    traj_frames[feat_key] = [init_frame]
                    traj_frames[f"{feat_key}_1_pred"] = []

            for etype in self.edge_types:
                split_sizes = g.batch_num_edges(etype).detach().cpu().tolist()
                init_frame = g.edges[etype].data["e_0"].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[etype] = [init_frame]
                traj_frames[f"{etype}_1_pred"] = []

        dst_dict = None
        for s_idx in range(1, t.shape[0]):
            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # compute next step and set x_t = x_s
            g, dst_dict = self.step(
                g,
                s_i,
                t_i,
                alpha_t_i,
                alpha_s_i,
                alpha_t_prime_i,
                node_batch_idx,
                upper_edge_mask,
                prev_dst_dict=dst_dict,
                **kwargs,
            )

            if visualize:
                for ntype in self.node_types:
                    split_sizes = g.batch_num_nodes(ntype).detach().cpu().tolist()

                    for feat in canonical_node_features[ntype]:
                        feat_key = f"{ntype}_{feat}"
                        frame = g.nodes[ntype].data[f"{feat}_t"].detach().cpu()
                        frame = torch.split(frame, split_sizes)
                        traj_frames[feat_key].append(frame)

                        ep_key = f"{feat_key}_1_pred"
                        if f"{feat}_1_pred" in g.nodes[ntype].data:
                            ep_frame = (
                                g.nodes[ntype].data[f"{feat}_1_pred"].detach().cpu()
                            )
                            ep_frame = torch.split(ep_frame, split_sizes)
                            traj_frames[ep_key].append(ep_frame)

                for etype in self.edge_types:
                    split_sizes = g.batch_num_edges(etype).detach().cpu().tolist()
                    frame = g.edges[etype].data["e_t"].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[etype].append(frame)

                    ep_key = f"{etype}_1_pred"
                    if "e_1_pred" in g.edges[etype].data:
                        ep_frame = g.edges[etype].data["e_1_pred"].detach().cpu()
                        ep_frame = torch.split(ep_frame, split_sizes)
                        traj_frames[ep_key].append(ep_frame)

        # set x_1 = x_t
        for ntype in self.node_types:
            for feat in canonical_node_features[ntype]:
                g.nodes[ntype].data[f"{feat}_1"] = g.nodes[ntype].data[f"{feat}_t"]

        for etype in self.edge_types:
            g.edges[etype].data["e_1"] = g.edges[etype].data["e_t"]

        if visualize:
            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for sys_idx in range(g.batch_size):
                system_dict = {}
                for ntype in self.node_types:
                    system_dict[ntype] = {}
                    for feat in canonical_node_features[ntype]:
                        feat_key = f"{ntype}_{feat}"
                        if feat_key in traj_frames:
                            feat_traj = []
                            n_frames = len(traj_frames[feat_key])
                            for frame_idx in range(n_frames):
                                if sys_idx < len(traj_frames[feat_key][frame_idx]):
                                    feat_traj.append(
                                        traj_frames[feat_key][frame_idx][sys_idx]
                                    )
                            if feat_traj:
                                system_dict[ntype][feat] = torch.stack(feat_traj)
                        pred_key = f"{feat_key}_1_pred"
                        if pred_key in traj_frames and traj_frames[pred_key]:
                            pred_traj = []
                            n_frames = len(traj_frames[pred_key])
                            for frame_idx in range(n_frames):
                                if sys_idx < len(traj_frames[pred_key][frame_idx]):
                                    pred_traj.append(
                                        traj_frames[pred_key][frame_idx][sys_idx]
                                    )
                            if pred_traj:
                                system_dict[ntype][f"{feat}_1_pred"] = torch.stack(
                                    pred_traj
                                )
                system_dict["edges"] = {}
                for etype in self.edge_types:
                    if etype in traj_frames:
                        edge_traj = []
                        n_frames = len(traj_frames[etype])
                        for frame_idx in range(n_frames):
                            if sys_idx < len(traj_frames[etype][frame_idx]):
                                edge_traj.append(traj_frames[etype][frame_idx][sys_idx])
                        if edge_traj:
                            system_dict["edges"][etype] = torch.stack(edge_traj)
                    pred_key = f"{etype}_1_pred"
                    if pred_key in traj_frames and traj_frames[pred_key]:
                        pred_traj = []
                        n_frames = len(traj_frames[pred_key])
                        for frame_idx in range(n_frames):
                            if sys_idx < len(traj_frames[pred_key][frame_idx]):
                                pred_traj.append(
                                    traj_frames[pred_key][frame_idx][sys_idx]
                                )
                        if pred_traj:
                            system_dict["edges"][f"{etype}_1_pred"] = torch.stack(
                                pred_traj
                            )
                reshaped_traj_frames.append(system_dict)
            return g, reshaped_traj_frames

        return g

    def step(
        self,
        g: dgl.DGLGraph,
        s_i: torch.Tensor,
        t_i: torch.Tensor,
        alpha_t_i: Dict[str, torch.Tensor],
        alpha_s_i: Dict[str, torch.Tensor],
        alpha_t_prime_i: Dict[str, torch.Tensor],
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
        prev_dst_dict: Dict,
        inv_temp_func=None,
        **kwargs,
    ):
        if inv_temp_func is None:
            inv_temp_func = self.continuous_inv_temp_func

        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            prev_dst_dict=prev_dst_dict,
        )

        # compute x_s for each feature and set x_t = x_s
        for ntype in self.node_types:
            for feat in canonical_node_features[ntype]:
                x_t = g.nodes[ntype].data[f"{feat}_t"]
                x_1 = dst_dict["nodes"][ntype][feat]

                # evaluate the vector field at the current timepoint
                vf = self.vector_field(
                    x_t,
                    x_1,
                    alpha_t_i[f"{ntype}_{feat}"],
                    alpha_t_prime_i[f"{ntype}_{feat}"],
                )

                # apply temperature scaling
                vf = vf * inv_temp_func(t_i)

                # apply euler integration step
                x_s = x_t + vf * (s_i - t_i)

                # record predicted endoint, for visualization purposes
                g.nodes[ntype].data[f"{feat}_1_pred"] = x_1.detach().clone()

                # record updated feature in the graph
                g.nodes[ntype].data[f"{feat}_t"] = x_s

        for etype in self.edge_types:
            x_t = g.edges[etype].data["e_t"]
            x_t = x_t[upper_edge_mask[etype]]

            x_1 = dst_dict["edges"][etype]

            vf = self.vector_field(x_t, x_1, alpha_t_i[etype], alpha_t_prime_i[etype])
            # apply temperature scaling
            vf = vf * inv_temp_func(t_i)

            # apply euler integration step
            x_s = x_t + vf * (s_i - t_i)

            # set the edge features so that corresponding upper and lower triangle edges have the same value
            e_s = torch.zeros_like(g.edges[etype].data["e_0"])
            e_s[upper_edge_mask[etype]] = x_s
            e_s[~upper_edge_mask[etype]] = x_s
            x_s = e_s

            e_1 = torch.zeros_like(g.edges[etype].data["e_0"])
            e_1[upper_edge_mask[etype]] = dst_dict["edges"][etype]
            e_1[~upper_edge_mask[etype]] = dst_dict["edges"][etype]
            x_1 = e_1

            # record predicted endoint, for visualization purposes
            g.edges[etype].data[f"e_1_pred"] = x_1.detach().clone()

            # record updated feature in the graph
            g.edges[etype].data[f"e_t"] = x_s

        return g, dst_dict

    def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime):
        vf = alpha_t_prime / (1 - alpha_t) * (x_1 - x_t)
        return vf


class NodePositionUpdate(nn.Module):
    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):
            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(
        self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor
    ):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class EdgeUpdate(nn.Module):
    def __init__(
        self,
        n_node_scalars,
        n_edge_feats,
        update_edge_w_distance=False,
        rbf_dim=16,
    ):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars * 2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
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
        ]

        if self.update_edge_w_distance and d is not None:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(
            edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1))
        )
        return edge_feats
