from mtl.datasets.transforms import TwoCropsTransform
from ..data_builder import DATASETS
from .emb_base import EmbBaseDataset


@DATASETS.register_module()
class SslPretrainMocoDataset(EmbBaseDataset):
    """SSL pretraining dataset"""

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        super(SslPretrainMocoDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

        self.augment_transform = TwoCropsTransform(self.pipeline)

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        if self.is_tfrecord:
            results = self.getitem_info(idx)
        else:
            results = self.getitem_info(idx, return_img=True)

        if self.test_mode:
            return self.pipeline(results)
        else:
            return self.augment_transform(results)
