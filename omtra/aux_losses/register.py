from typing import Dict


AUX_LOSS_REGISTER = {}

def register_aux_loss(name: str):
    """
    Decorator to register an auxiliary loss class with a given name.
    Also attaches the key to the class as an attribute.
    :param name: A unique name for the auxiliary loss.
    """
    def decorator(cls):
        cls.name = name  # Attach the key to the class.
        AUX_LOSS_REGISTER[name] = cls
        return cls
    return decorator


def aux_loss_name_to_class(name: str):
    """
    Get the auxiliary loss class associated with a given name.
    :param name: The name of the auxiliary loss.
    :return: The auxiliary loss class associated with the name.
    """
    return AUX_LOSS_REGISTER[name]
