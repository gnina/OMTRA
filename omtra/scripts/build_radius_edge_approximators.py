import torch_cluster as tc
import dgl
from omtra.data.graph.utils import get_batch_idxs
from omtra.tasks.utils import get_edges_for_task
from omtra.data.graph import edge_types as all_edge_types
from omtra.data.graph import to_canonical_etype
from omtra.data.graph.utils import get_edges_per_batch
import torch
from pathlib import Path
import pandas as pd 
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from omtra.load.quick import load_cfg, datamodule_from_config
from omtra.utils import omtra_root
import argparse

def parse_args(): 
    p = argparse.ArgumentParser()
    p.add_argument('--r_min', type=float, default=2.0)
    p.add_argument('--r_max', type=float, default=12.0)
    p.add_argument('--n_r', type=int, default=15)
    p.add_argument('--n_batches', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--n_train_steps', type=int, default=30000)
    p.add_argument('--output_dir', type=str, default=None)
    args = p.parse_args()
    return args

def get_training_data(r_min, r_max, n_r, n_batches, batch_size, plinder_dataset):
    rows = []
    task = 'protein_ligand_pharmacophore_denovo'
    radii = torch.linspace(r_min, r_max, n_r)
    for i in range(n_batches):
        gs = []
        start_idx = torch.randint(0, len(plinder_dataset) - batch_size, (1,)).item()
        for i in range(batch_size):
            idx = (task, i+start_idx)
            g = plinder_dataset[idx]
            gs.append(g)
        g = dgl.batch(gs)
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)


        for etype in all_edge_types:

            src_ntype, _, dst_ntype = to_canonical_etype(etype)
            
            if 'covalent' in etype:
                continue

            if src_ntype == 'prot_res' or dst_ntype == 'prot_res':
                continue

            if g.num_nodes(src_ntype) == 0 or g.num_nodes(dst_ntype) == 0:
                continue

            src_pos = g.nodes[src_ntype].data['x_1_true']
            dst_pos = g.nodes[dst_ntype].data['x_1_true']
            # if len(src_pos.shape) == 3:
                # continue

            for r in radii:
                edge_idxs = tc.radius(x=dst_pos, y=src_pos, r=r.item(), batch_x=node_batch_idxs[dst_ntype], batch_y=node_batch_idxs[src_ntype])
                edges_per_g = get_edges_per_batch(edge_idxs[0], g.batch_size, node_batch_idxs[src_ntype])
                n_dst_nodes = g.batch_num_nodes(dst_ntype)
                for i, (n_nodes, n_edges) in enumerate(zip(g.batch_num_nodes(src_ntype), edges_per_g)):
                    row = {
                        'etype': etype,
                        'src_ntype': src_ntype,
                        'dst_ntype': dst_ntype,
                        'r': r.item(),
                        'n_src_nodes': int(n_nodes),
                        'n_dst_nodes': n_dst_nodes[i].item(),
                        'edges_per_node': int(n_edges)/(int(n_nodes)+1e-3),
                        'n_edges': int(n_edges),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_data_for_mlp(df_etype, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    x = df_etype[['r', 'n_src_nodes', 'n_dst_nodes']].values
    y = df_etype['edges_per_node'].values
    x_gpu = torch.from_numpy(x).float().to(device)
    y_gpu = torch.from_numpy(y).float().to(device)
    return x, y, x_gpu, y_gpu

def get_df_etype(df, etype):
    df_etype = df[df['etype'] == etype]
    return df_etype

def train_edge_mlp(etype, df_etype, device='cuda:0' if torch.cuda.is_available() else 'cpu', n_steps=15000):

    mlp = nn.Sequential(
        nn.Linear(3, 12),
        nn.SiLU(),
        nn.Linear(12, 12),
        nn.SiLU(),
        nn.Linear(12, 1),
    ).to(device)

    x, y, x_gpu, y_gpu = get_data_for_mlp(df_etype, device=device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.5e-3)

    losses = []
    for i in range(n_steps):
        optimizer.zero_grad()
        y_hat = mlp(x_gpu)
        l = loss(y_hat.squeeze(), y_gpu)
        l.backward()
        optimizer.step()

        losses.append(l.item())

    return mlp, losses

def plot_cdf(metric, metric_name, ax=None):

    if ax is None:
        ax = plt.gca()

    # Sort the metric array
    sorted_metric = np.sort(metric)

    # Generate the cumulative distribution values
    cdf = np.arange(1, len(sorted_metric) + 1) / len(sorted_metric)

    # Plot the CDF
    ax.plot(sorted_metric, cdf, label=f'CDF of {metric_name}')
    ax.set_xlabel(f'{metric_name}')
    ax.set_ylabel('CDF')
    ax.set_title(f'CDF of {metric_name}')
    ax.grid()

@torch.no_grad()
def evaluate_model(mlp, df_etype, device='cuda:0' if torch.cuda.is_available() else 'cpu'):

    x, y, x_gpu, y_gpu = get_data_for_mlp(df_etype, device=device)

    y_hat = mlp(x_gpu).squeeze(-1)

    num_edges_true = x_gpu[:, 1]*y_gpu
    num_edges_pred = x_gpu[:, 1]*y_hat
    n_edges_err = num_edges_pred - num_edges_true

    num_edges_true = num_edges_true.detach().cpu().numpy()
    n_edges_err = n_edges_err.detach().cpu().numpy()
    edge_error_frac = np.abs(n_edges_err)/(num_edges_true+1)
    y_hat = y_hat.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    sc = axes[0,0].scatter(y, y_hat, c=n_edges_err, cmap='bwr', s=50, edgecolor='k')
    cbar = fig.colorbar(sc, ax=axes[0,0])
    cbar.set_label('Error in number of edges')

    # get xlim and ylim
    xlim = axes[0,0].get_xlim()
    ylim = axes[0,0].get_ylim()

    # plot y=x line over the current x and y limits
    x = torch.linspace(xlim[0], xlim[1], 100)
    axes[0,0].plot(x.cpu(), x.cpu(), color='red', linestyle='--', lw=3)
    axes[0,0].set_xlabel('True Edges per Node')
    axes[0,0].set_ylabel('Predicted Edges per Node')

    axes[0, 1].hist(edge_error_frac, bins=50)
    axes[0, 1].set_xlabel('Edge Error Fraction')
    # axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Histogram of Edge Error Fraction')

    axes[1, 0].hist(n_edges_err, bins=50)
    axes[1, 0].set_xlabel('Edge Error')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Histogram of Edge Error')

    plot_cdf(edge_error_frac, 'Edge Error Fraction', ax=axes[1, 1])
    plot_cdf(np.abs(n_edges_err), 'Edge Error', ax=axes[1, 2])

    axes[0, 2].axis('off')

    fig.subplots_adjust(
        hspace=0.4,   # vertical space between rows
        wspace=0.4    # horizontal space between columns
    )

    fig.suptitle(f'N edge approximation for {etype}', fontsize=16)


if __name__ == "__main__":

    args = parse_args()

    if args.output_dir is None:
        args.output_dir = Path(omtra_root()) / 'omtra' / 'constants' / 'radius_edge_approximators'
    else:
        args.output_dir = Path(args.output_dir)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f'loading dataset object')
    cfg = load_cfg()
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("train")
    plinder_dataset = train_dataset.datasets['plinder']['no_links']

    print('extracting radius graphs')
    df = get_training_data(args.r_min, args.r_max, args.n_r, args.n_batches, args.batch_size, plinder_dataset)

    # print('training predictors')
    edge_pred_mlps = {}
    for etype, df_etype in tqdm(df.groupby('etype'), desc='Training edge predictors', total=len(df['etype'].unique())):
        mlp, losses = train_edge_mlp(etype, df_etype, n_steps=args.n_train_steps)
        edge_pred_mlps[etype] = mlp


    for etype in edge_pred_mlps:
        evaluate_model(edge_pred_mlps[etype], get_df_etype(df, etype))
        plt.savefig(args.output_dir / f'{etype}.png', bbox_inches='tight')
        plt.close()

    for k in edge_pred_mlps:
        edge_pred_mlps[k] = edge_pred_mlps[k].cpu()

    torch.save(edge_pred_mlps, args.output_dir / 'edge_pred_mlps.pt')



