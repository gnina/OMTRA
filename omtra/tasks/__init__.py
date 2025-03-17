# tasks/__init__.py

# Import the registry module first so that TASK_REGISTRY is initialized.
import omtra.tasks.register as register


# Dynamically import all modules in the tasks package.
# (This approach automatically imports any new modules you add to the package.)
import os
import pkgutil
import importlib

# Get the directory of the current package.
package_dir = os.path.dirname(__file__)

# Iterate through modules in the package.
for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
    # Optionally, skip certain modules (like 'registry' or '__init__').
    if module_name in {"register", "__init__"}:
        continue
    importlib.import_module(f"{__name__}.{module_name}")