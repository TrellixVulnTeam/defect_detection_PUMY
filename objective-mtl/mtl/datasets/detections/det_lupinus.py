from mtl.datasets.data_wrapper import DATASETS
from .det_base import DetBaseDataset


@DATASETS.register_module()
class DetLupinusDataset(DetBaseDataset):
    """Data interface for the lupinus dataset"""

    class_names = None

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Same as base detection dataset"""
        super(DetLupinusDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        self.cat2label = {cat: i for i, cat in enumerate(self.class_names)}
