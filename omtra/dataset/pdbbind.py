from omtra.dataset.plinder import PlinderDataset
from omtra.utils.misc import classproperty


class PdbbindDataset(PlinderDataset):
    """PDBBind DynamicBind splits stored in Plinder-compatible pred zarr format."""

    @classproperty
    def name(cls) -> str:
        return "pdbbind"
