from ..data_builder import DATASETS
from .emb_base import EmbBaseDataset


@DATASETS.register_module()
class SslPretrainMaeDataset(EmbBaseDataset):
    """SSL pretraining dataset"""

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        super(SslPretrainMaeDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )
