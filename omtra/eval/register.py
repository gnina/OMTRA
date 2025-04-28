

eval_register = {}

# registers map task name to list of eval functions


def register_eval(name: str):
    """
    Decorator to register evaluation function with a given task name.
    :param name: A unique name for the task.
    """

    def decorator(fn):
        fn.name = name  # Attach the key to the class.
        eval_register[name] = fn
        return fn

    return decorator



def get_eval(name: str):
    """
    Get the eval function with the given task name.
    :param name: The name of the task.
    :return: The eval functions.
    """
    return eval_register[name]

