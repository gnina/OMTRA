from typing import Optional
import torch
from torch.types import Device
import dgl

# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


# the following is OG content

def center_on_ligand_gt(g: dgl.DGLHeteroGraph):
    lig_com = g.nodes['lig'].data['x_1_true'].mean(dim=0, keepdim=True)
    for ntype in g.ntypes:
        if 'x_1_true' not in g.nodes[ntype].data:
            continue
        g.nodes[ntype].data['x_1_true'] -= lig_com
    return g

def rotate_ground_truth(g: dgl.DGLHeteroGraph):

    device = g.device
    dtype = g.nodes[g.ntypes[0]].data['x_1_true'].dtype
    
    # random rotation matrix
    R = random_rotations(1,
        dtype=dtype,
        device=device).squeeze(0)

    for ntype in g.ntypes:
        if 'x_1_true' not in g.nodes[ntype].data:
            continue

        # Apply rotation
        x_rotated = g.nodes[ntype].data['x_1_true'] @ R.T
        
        # Update the graph feature
        g.nodes[ntype].data['x_1_true'] = x_rotated
    return g

def system_offset(g: dgl.DGLHeteroGraph, offset_std: float = 0.0):

    if offset_std <= 0.0:
        return g

    offset = torch.randn(1,3, device=g.device)*offset_std
    for ntype in g.ntypes:
        if 'x_1_true' not in g.nodes[ntype].data:
            continue
        g.nodes[ntype].data['x_1_true'] += offset
    return g