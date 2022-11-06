from ..data_builder import DATASETS
from .cls_base import ClsBaseDataset


@DATASETS.register_module()
class ClsLupinusDataset(ClsBaseDataset):
    """Cls imagenet classification"""

    class_names = None

    def evaluate(self, results, metric="accuracy", metric_options=None, logger=None):
        if metric_options is None:
            num_classes = len(self.class_names)
            if num_classes < 10:
                num_topk = num_classes // 2
            else:
                num_topk = 5
            metric_options = {"topk": (1, num_topk)}

        return super(ClsLupinusDataset, self).evaluate(
            results, metric, metric_options, logger
        )
