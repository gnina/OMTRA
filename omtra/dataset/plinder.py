import dgl
import torch

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task

class PlinderDataset(ZarrDataset):
    def __init__(self, 
                 zarr_store_path: str,
                 graphs_per_chunk: int
    ):
        super().__init__(zarr_store_path)
        self.graphs_per_chunk = graphs_per_chunk

    @classmethod
    def name(cls):
        return 'plinder'

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class[task_name]

        # TODO: things!
        g = build_complex_graph()

        return g