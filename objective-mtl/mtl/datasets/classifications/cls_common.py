from ..data_builder import DATASETS
from .cls_base import ClsBaseDataset


@DATASETS.register_module()
class ClsCommonDataset(ClsBaseDataset):
    """Cls common classification"""

    class_names = [
        "level-1",
        "level-2",
        "level-3",
        "level-4",
        "level-5",
        "level-6",
        "level-7",
        "level-8",
        "level-9",
        "level-10",
    ]

    def load_annotations(self):
        """Load data_infos"""
        data_infos = []
        with open(self.ann_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            tmp_str = line.strip()
            str_list = tmp_str.split()
            if len(str_list) >= 2:
                label = int(str_list[0])
                file_name = str_list[1]

                data_infos.append(dict(file_name=file_name, label=label))

        return data_infos

    def evaluate(self, results, metric="accuracy", metric_options=None, logger=None):
        if metric_options is None:
            metric_options = {"topk": (1, 3)}
        return super(ClsCommonDataset, self).evaluate(
            results, metric, metric_options, logger
        )
