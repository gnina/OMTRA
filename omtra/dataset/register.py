from omtra.dataset.pharmit import PharmitDataset
from omtra.dataset.plinder import PlinderDataset
from omtra.dataset.crossdocked import CrossdockedDataset

supported_datasets = [
    PharmitDataset,
    PlinderDataset, 
    CrossdockedDataset,
]

dataset_names = [cls_obj.name for cls_obj in supported_datasets]
assert len(set(dataset_names)) == len(dataset_names), "Duplicate dataset names found!"

dataset_name_to_class: dict = dict(zip(dataset_names, supported_datasets))

def display_datasets():
    for dataset_class, dataset_name in zip(supported_datasets, dataset_names):
        print(f"Dataset class: {dataset_class}")
        print(f"Dataset name: {dataset_name}")
        print()