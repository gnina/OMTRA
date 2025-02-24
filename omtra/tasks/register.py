import inspect
import omtra.tasks.tasks as tasks
from typing import Dict

task_classes = [cls_obj 
                for cls_name, cls_obj in inspect.getmembers(tasks, inspect.isclass) 
                if issubclass(cls_obj, tasks.Task) and cls_obj is not tasks.Task]

task_name_to_class: Dict[str, tasks.Task] = {cls_obj.name: cls_obj for cls_obj in task_classes}

def display_tasks():
    task_names = [cls_obj.name for cls_obj in task_classes]
    for task_class, task_name in zip(task_classes, task_names):
        print(f"Task class: {task_class}")
        print(f"Task name: {task_name}")
        print()