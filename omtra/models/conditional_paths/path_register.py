from typing import Dict, Callable

COND_PATH_REGISTER = Dict[str, Callable] = {}

def register_conditional_path(name: str):
    """
    Decorator to register a conditional path function with a given name.
    :param name: A unique name for the conditional path.
    """
    def decorator(fn):
        COND_PATH_REGISTER[name] = fn
        return fn
    return decorator