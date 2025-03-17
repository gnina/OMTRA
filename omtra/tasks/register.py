from typing import Dict


TASK_REGISTER = {}

def register_task(name: str):
    """
    Decorator to register a task class with a given name.
    Also attaches the key to the class as an attribute.
    :param name: A unique name for the task.
    """
    def decorator(cls):
        cls.name = name  # Attach the key to the class.
        TASK_REGISTER[name] = cls
        return cls
    return decorator


def task_name_to_class(name: str):
    """
    Get the task class associated with a given name.
    :param name: The name of the task.
    :return: The task class associated with the name.
    """
    return TASK_REGISTER[name]


def display_tasks():
    for task_name, task_class in TASK_REGISTER.items():
        print(f"Task class: {task_class}")
        print(f"Task name: {task_name}")
        print()