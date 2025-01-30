import torch
from abc import ABC, abstractmethod

class OMTRADataset(ABC, torch.utils.data.Dataset):
    """Base class for single datasets."""

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass