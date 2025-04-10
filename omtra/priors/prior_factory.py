# prior_factory.py
from omtra.priors.register import train_prior_register, inference_prior_register
from functools import partial

def get_prior(task_cls, config_prior=None, training=False) -> dict:
    """
    Get the prior distribution function for all modalitities for a given task class.
    :param task_cls: The task class (e.g., TaskA)
    :param config_prior: Optional config override dict.
    :return: A dictionary with keys for each modality and values that are the prior distribution functions.
    """
    if training:
        register = train_prior_register
    else:
        register = inference_prior_register

    task_name = task_cls.name
    prior_fn_output = {}
    for modality in task_cls.modalities_generated:
        try:
            prior_fn_key = config_prior[task_name][modality.name]['type']
        except (KeyError, TypeError): # NOTE: changed to handle NoneType config_prior
            prior_fn_key = task_cls.priors[modality.name]['type']

        try:
            prior_params = config_prior[task_name][modality.name]['params']
        except (KeyError, TypeError): # NOTE: changed to handle NoneType config_prior
            prior_params = task_cls.priors[modality.name].get('params', {})

        prior_fn = register[prior_fn_key]
        prior_fn = partial(prior_fn, **prior_params)
        prior_fn_output[modality.name] = (prior_fn_key, prior_fn)

    return prior_fn_output