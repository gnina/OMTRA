from collections import defaultdict

train_eval_register = defaultdict(list)
inference_eval_register = defaultdict(list)

# registers map task name to list of eval functions


def register_train_eval(name: str):
    """
    Decorator to register training time eval function with a given name.
    :param name: A unique name for the prior.
    """

    def decorator(fn):
        fn.name = name  # Attach the key to the class.
        train_eval_register[name].append(fn())
        return fn

    return decorator


def register_inference_eval(name: str):
    """
    Decorator to register an inference time eval function with a given name.
    :param name: A unique name for the task.
    """

    def decorator(fn):
        fn.name = name  # Attach the key to the class.
        inference_eval_register[name].append(fn())
        return fn

    return decorator


def get_train_eval(name: str):
    """
    Get the training time eval functions with the given name.
    :param name: The name of the eval.
    :return: The eval functions.
    """
    return train_eval_register[name]


def get_inference_eval(name: str):
    """
    Get the inference time eval functions with the given name.
    :param name: The name of the eval.
    :return: The eval functions.
    """
    return inference_eval_register[name]
