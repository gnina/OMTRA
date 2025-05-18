# eval_factory.py
from omtra.eval.register import eval_register
import peppr


def get_evaluator(task_name: str):
    """
    Get the eval functions for a given task class.
    :param task_cls: The task class (e.g., TaskA)
    :return: A peppr Evaluator object.
    """

    evals = eval_register.get(task_name)
    if not evals:
        raise NotImplementedError(f"No evals registered for task {task_name}.")

    evaluator = peppr.Evaluator(evals)

    return evaluator
