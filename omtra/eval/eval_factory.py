# eval_factory.py
from omtra.eval.register import train_eval_register, inference_eval_register
import peppr


def get_evaluator(task_name: str, training: bool = False):
    """
    Get the eval functions for a given task class.
    :param task_cls: The task class (e.g., TaskA)
    :return: A peppr Evaluator object.
    """
    type = "training"
    if training:
        register = train_eval_register
    else:
        register = inference_eval_register
        type = "inference"

    evals = register.get(task_name)
    if not evals:
        raise NotImplementedError(f"No {type} evals registered for task {task_name}.")

    evaluator = peppr.Evaluator(evals)

    return evaluator
