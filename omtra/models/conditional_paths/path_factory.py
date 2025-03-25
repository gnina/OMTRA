from omtra.models.conditional_paths.path_register import condpath_name_to_fn
from omegaconf import DictConfig
from functools import partial
from typing import Dict, Tuple, Callable


def get_conditional_path_fns(
    task_cls, conditional_path_config: DictConfig
) -> Dict[str, Tuple[str, Callable]]:
    """
    Get the conditional path functions for a given task class.
    :param
    task_cls: The task class (e.g., TaskA)
    :param
    conditional_path_config: The conditional path configuration.
    :return: A dictionary with keys for each modality and values that are the conditional path functions.
    """
    conditional_path_fn_output = {}
    for modality in task_cls.modalities_generated:
        try:
            conditional_path_fn_key = conditional_path_config[task_cls.name][
                modality.name
            ]["type"]
        except KeyError:
            conditional_path_fn_key = task_cls.conditional_paths[modality.name]["type"]

        try:
            conditional_path_params = conditional_path_config[task_cls.name][
                modality.name
            ]["params"]
        except KeyError:
            conditional_path_params = task_cls.conditional_paths[modality.name].get(
                "params", {}
            )

        conditional_path_fn = condpath_name_to_fn(conditional_path_fn_key)
        conditional_path_fn = partial(conditional_path_fn, **conditional_path_params)
        conditional_path_fn_output[modality.name] = (
            conditional_path_fn_key,
            conditional_path_fn,
        )

    return conditional_path_fn_output
