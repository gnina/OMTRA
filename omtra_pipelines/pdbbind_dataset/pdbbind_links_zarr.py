"""PDBBind zarr converter using manifest-driven file paths."""

import logging
from typing import Optional

import pandas as pd

from omtra_pipelines.pdbbind_dataset.pdbbind_pipeline import PdbbindSystemProcessor
from omtra_pipelines.plinder_dataset.plinder_links_zarr import PlinderLinksZarrConverter
from omtra_pipelines.plinder_dataset.utils import setup_logger

logger = setup_logger(__name__)


class PdbbindLinksZarrConverter(PlinderLinksZarrConverter):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        raw_data_root: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.manifest_by_id = manifest_df.set_index("system_id")
        self.raw_data_root = raw_data_root

    def _process_system(self, system_id: str):
        try:
            if system_id not in self.manifest_by_id.index:
                logger.warning(f"System {system_id} not in manifest")
                return None
            row = self.manifest_by_id.loc[system_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if not row.get("files_ok", True):
                logger.warning(f"Skipping {system_id}: files missing on disk")
                return None

            processor = PdbbindSystemProcessor(
                system_id=system_id,
                holo_pdb=row["holo_pdb"],
                ligand_sdf=row["ligand_sdf"],
                pred_pdb=row["pred_pdb"],
                link_id=row["uid"],
                pocket_cutoff=self.pocket_cutoff,
                raw_data_root=self.raw_data_root,
            )
            return processor.process_system()
        except Exception as exc:
            logging.exception(f"Error processing system {system_id}: {exc}")
            return None
