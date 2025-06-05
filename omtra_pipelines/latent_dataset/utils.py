import torch
from omtra.eval.system import SampledSystem

# this implementation is from: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8?permalink_comment_id=5286981#gistcomment-5286981
# as a sanity check we can compare it to the implementation in rdkit:
def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -     aligned : (N, D) torch.Tensor --  The coordinates of `A` after the optimal rotation/translation

    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> A_aligned = find_rigid_alignment(A, B)
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T

    pred_aligned = (R.mm(A.T)).T + t.squeeze()

    return pred_aligned

def get_gt_as_rdkit_ligand(g):
    """
        Used to wrap a ground truth graph as a SampledSystem, so that we 
        can call get_rdkit_ligand() on it.
    """
    gt_graph_for_rdkit = g.clone()
    
    lig_node_data = gt_graph_for_rdkit.nodes['lig'].data
    if 'x_1_true' in lig_node_data:
        lig_node_data['x_1'] = lig_node_data['x_1_true']
    if 'a_1_true' in lig_node_data:
        lig_node_data['a_1'] = lig_node_data['a_1_true']
    if 'c_1_true' in lig_node_data:
        lig_node_data['c_1'] = lig_node_data['c_1_true']
        
    lig_edge_data = gt_graph_for_rdkit.edges["lig_to_lig"].data
    lig_edge_data['e_1'] = lig_edge_data['e_1_true']
    
    gt_system_wrapper = SampledSystem(g=gt_graph_for_rdkit.to('cpu'))
    
    return gt_system_wrapper.get_rdkit_ligand()