import torch

def flow_matching_ood_loss(mu, log_sigma2, x_t, lam=1.0):
    """
    DrugFlow FM-OOD loss from Equation (1):
    L = d/2 * log(sigma^2) + (1 / (2 * sigma^2)) * ||mu - target||^2
        + (λ/2) * (sigma^2 - 1)^2
    """
    d = mu.shape[-1]  # dimensionality 
    sigma2 = torch.exp(log_sigma2)  # ensure positivity

    mse_term = ((mu - x_t) ** 2).sum(dim=-1)  # sum over vector components
    log_term = (d / 2.0) * log_sigma2.squeeze(-1)
    nll_term = 0.5 * mse_term / sigma2.squeeze(-1)
    reg_term = 0.5 * lam * ((sigma2.squeeze(-1) - 1.0) ** 2)

    per_atom_loss = (log_term + nll_term + reg_term)
    return per_atom_loss.mean(), per_atom_loss, log_term.mean(), nll_term.mean(), reg_term.mean() 
