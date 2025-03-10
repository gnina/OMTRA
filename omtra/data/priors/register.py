from omtra.data.priors.priors import *

train_prior_register = {
    'centered-normal': centered_normal_prior,
    'gaussian': gaussian,
    'masked': ctmc_masked_prior
}

inference_prior_register = {
    'centered-normal': centered_normal_prior_batched_graph,
    'gaussian': gaussian,
    'masked': ctmc_masked_prior
}