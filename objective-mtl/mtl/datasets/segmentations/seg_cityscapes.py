import os
import os.path as osp
import tempfile
import numpy as np
from PIL import Image

from mtl.utils.log_util import print_log
from mtl.datasets.data_wrapper import DATASETS
from mtl.utils.path_util import mkdir_or_exist
from mtl.utils.misc_util import ProgressBar
from .seg_base import SegBaseDataset


@DATASETS.register_module()
class CityscapesDataset(SegBaseDataset):
    """Cityscapes dataset."""

    class_names = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
    palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        super(CityscapesDataset, self).__init__(
            data_cfg, pipeline_cfg, root_path, sel_index
        )

    def get_annotations(self, data_cfg, sel_index):
        if "DATA_PREFIX" in data_cfg and isinstance(data_cfg.DATA_PREFIX, list):
            self.data_prefix = data_cfg.DATA_PREFIX[sel_index]
        else:
            self.data_prefix = None
        if "ANNO_PREFIX" in data_cfg and isinstance(data_cfg.ANNO_PREFIX, list):
            self.anno_prefix = data_cfg.ANNO_PREFIX[sel_index]
        else:
            self.anno_prefix = None
        if "SEG_PREFIX" in data_cfg and isinstance(data_cfg.SEG_PREFIX, list):
            self.seg_prefix = data_cfg.SEG_PREFIX[sel_index]
        else:
            self.seg_prefix = None
        if not (self.data_prefix is None or osp.isabs(self.data_prefix)):
            self.data_prefix = osp.join(self.data_root, self.data_prefix)
        if not (self.anno_prefix is None or osp.isabs(self.anno_prefix)):
            self.anno_prefix = osp.join(self.data_root, self.anno_prefix)
        if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
            self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        if self.ann_file == "":
            self.ann_file = None
        else:
            self.ann_file = self.ann_file[0]
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
        self.data_suffix = "_leftImg8bit.png"
        self.anno_suffix = "_gtFine_labelTrainIds.png"
        self.data_infos = self.load_annotations()

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels

        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.data_infos[idx]["file_name"]
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f"{basename}.png")

            output = Image.fromarray(result.astype(np.uint8)).convert("P")
            import cityscapesscripts.helpers.labels as CSLabels

            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), (
            "The length of results is not equal to the dataset len: "
            f"{len(results)} != {len(self)}"
        )

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric="mIoU",
        logger=None,
        imgfile_prefix=None,
        efficient_test=False,
    ):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """
        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if "cityscapes" in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix)
            )
            metrics.remove("cityscapes")
        if len(metrics) > 0:
            eval_results.update(
                super(CityscapesDataset, self).evaluate(
                    results, metrics, logger, efficient_test
                )
            )

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError(
                'Please run "pip install cityscapesscripts" to '
                "install cityscapesscripts first."
            )
        msg = "Evaluating in Cityscapes style"
        if logger is None:
            msg = "\n" + msg
        print_log(msg, logger=logger)

        _, tmp_dir = self.format_results(results, imgfile_prefix)

        if tmp_dir is None:
            result_dir = imgfile_prefix
        else:
            result_dir = tmp_dir.name

        eval_results = dict()
        print_log(f"Evaluating results under {result_dir} ...", logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in os.listdir(self.anno_prefix):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args)
        )

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results
