import struct
import os.path as osp
import random
from torch.utils.data import Dataset
from yacs.config import CfgNode
import numpy as np
from PIL import Image

from mtl.utils.io_util import CacheFiles
from mtl.utils.gen_tfrecords.wy_example_pb2 import Example
from mtl.utils.tfrecord_util import tfrecord2idx
from .transforms import Compose


class DataBaseDataset(Dataset):
    """Base dataset."""

    def __init__(self, data_cfg, pipeline_cfg, root_path, sel_index=0):
        """Initialization for dataset construction.

        Args:
            data_prefix (str): the prefix of data path
            pipeline (list): a list of dict, where each element represents
                a operation defined in transforms.
            ann_file (str | None): the annotation file. When ann_file is str,
                the subclass is expected to read from the ann_file. When ann_file
                is None, the subclass is expected to read according to data_prefix
            test_mode (bool): in train mode or test mode
        """
        if not isinstance(data_cfg, CfgNode):
            raise TypeError("data_cfg must be a list")
        if not isinstance(pipeline_cfg, CfgNode):
            raise TypeError("pipeline_cfg must be a list")

        self.data_root = root_path
        self.data_cfg = data_cfg

        if "DATA_INFO" in data_cfg and isinstance(data_cfg.DATA_INFO, list):
            self.ann_file = data_cfg.DATA_INFO[sel_index]
        else:
            raise ValueError("DATA_INFO should be set properly")

        if "VAL_MODE" in data_cfg:
            self.val_mode = data_cfg.VAL_MODE
        else:
            self.val_mode = False

        if "TEST_MODE" in data_cfg:
            self.test_mode = data_cfg.TEST_MODE
        else:
            self.test_mode = False

        self.is_tfrecord = data_cfg.IS_TFRECORD
        # processing pipeline
        self.pipeline_cfg = pipeline_cfg
        self.pipeline = Compose(self.get_pipeline_list())
        self.sel_index = sel_index
        self.sample_rate = (
            data_cfg.SAMPLE_RATE if hasattr(data_cfg, "SAMPLE_RATE") else 1.0
        )
        self.data_seed = data_cfg.DATA_SEED if hasattr(data_cfg, "DATA_SEED") else 7

        self.use_cache = data_cfg.USE_CACHE if hasattr(data_cfg, "USE_CACHE") else False
        self.cache_num_proc = (
            data_cfg.CACHE_NUM_PROC if hasattr(data_cfg, "CACHE_NUM_PROC") else 0
        )
        self.cache_dir = data_cfg.CACHE_DIR if hasattr(data_cfg, "CACHE_DIR") else ""

        if self.is_tfrecord:
            self.get_tfrecords()
        else:
            self.get_annotations(data_cfg, sel_index)

        if self.sample_rate < 0.999:
            data_len = self.samples if self.is_tfrecord else len(self.data_infos)
            sample_num = int(data_len * self.sample_rate)
            random.seed(self.data_seed)
            self.convert_idx = random.sample(list(range(data_len)), sample_num)
            self.samples = sample_num
            print(
                "DataSet: sample_rate={}, samples={}".format(
                    self.sample_rate, self.samples
                )
            )

        # set group flag for the sampler
        self._set_group_flag()

    def get_tfrecords(self):
        if self.use_cache:
            self.cache_files = CacheFiles(self.cache_dir, num_proc=self.cache_num_proc)
            self.cache_files.cache(self.ann_file)

        self.data_prefix = None
        tfindexs = []
        for i in range(len(self.ann_file)):
            if not osp.isabs(self.ann_file[i]):
                self.ann_file[i] = osp.join(self.data_root, self.ann_file[i])

            tfindexs.append(
                tfrecord2idx(
                    self.ann_file[i], self.ann_file[i].replace(".tfrecord", ".idx")
                )
            )
        self.data_infos = []
        self.tffiles = None
        self.samples = 0
        for index, tffile in zip(tfindexs, self.ann_file):
            num_count = 0
            with open(index) as idxf:
                for line in idxf:
                    offset, _ = line.split(" ")
                    self.data_infos.append((tffile, int(offset)))
                    num_count += 1
            self.samples += num_count
            print("load %s, samples:%s" % (tffile, num_count))

    def get_pipeline_list(self):
        """get the list of pipelines"""
        pipeline_list = []
        for k_t, v_t in self.pipeline_cfg.items():
            pipeline_item = {}
            if len(v_t) > 0:
                if not isinstance(v_t, CfgNode):
                    raise TypeError("pipeline items must be a CfgNode")
            pipeline_item["type"] = k_t

            for k_a, v_a in v_t.items():
                if isinstance(v_a, CfgNode):
                    if "type" in v_a:
                        pipeline_item[k_a] = {}
                        for sub_kt, sub_vt in v_a.items():
                            pipeline_item[k_a][sub_kt] = sub_vt
                    else:
                        pipeline_item[k_a] = []
                        for sub_kt, sub_vt in v_a.items():
                            sub_item = {}
                            if len(sub_vt) > 0:
                                if not isinstance(sub_vt, CfgNode):
                                    raise TypeError("transform items must be a CfgNode")
                            sub_item["type"] = sub_kt
                            for sub_ka, sub_va in sub_vt.items():
                                if isinstance(sub_va, CfgNode):
                                    raise TypeError("Only support two built-in layers")
                                sub_item[sub_ka] = sub_va
                            pipeline_item[k_a].append(sub_item)
                else:
                    pipeline_item[k_a] = v_a
            pipeline_list.append(pipeline_item)

        return pipeline_list

    def get_annotations(self, data_cfg, sel_index):
        if "DATA_PREFIX" in data_cfg and isinstance(data_cfg.DATA_PREFIX, list):
            self.data_prefix = data_cfg.DATA_PREFIX[sel_index]
        else:
            self.data_prefix = None

        # only use ann_file[0]
        self.ann_file = self.ann_file[0]
        # join paths if data_root is specified
        if not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.data_root, self.ann_file)
        if not (self.data_prefix is None or osp.isabs(self.data_prefix)):
            self.data_prefix = osp.join(self.data_root, self.data_prefix)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        raise NotImplementedError("Must Implement load_annotations")

    def record_parser(self, feature_list, return_img=True):
        """Call when is_tfrecord is ture."""
        raise NotImplementedError("Must Implement parser")

    def get_record(self, f, offset, return_img=True):
        """Get the record when is_tfrecord is true."""
        if not self.is_tfrecord:
            raise ValueError(
                "Please set is_tfrecord to be true when call this function"
            )
        f.seek(offset)
        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack("Q", byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) < proto_len:
            print(
                "read pb_data err,proto_len:%s pb_data len:%s"
                % (proto_len, len(pb_data))
            )
            return None
        example = Example()
        example.ParseFromString(pb_data)
        # keep key value in order
        feature = sorted(example.features.feature.items())
        record = self.record_parser(feature, return_img)
        # print(record)
        return record

    def prepare_data(self, idx):
        """Prepare data and run pipelines"""
        results = self.getitem_info(idx)
        return self.pipeline(results)

    def __len__(self):
        if self.is_tfrecord:
            return self.samples
        else:
            if self.sample_rate < 0.999:
                return len(self.convert_idx)
            else:
                return len(self.data_infos)

    def getitem_info(self, index, return_img=True):
        if self.is_tfrecord:
            # if self.tffiles is None:
            #     self.tffiles = dict()
            #     for idx, tffile in self.data_infos:
            #         self.tffiles[tffile] = open(tffile, "rb")
            tffile, offset = self.data_infos[index]
            f = open(tffile, "rb")
            item_info = self.get_record(f, offset, return_img)
            f.close()
            return item_info
        else:
            if not return_img:
                return self.data_infos[index]
            else:
                if "img" in self.data_infos[index]:
                    return self.data_infos[index]
                if self.data_prefix is not None:
                    img_path = osp.join(
                        self.data_prefix, self.data_infos[index]["file_name"]
                    )
                else:
                    img_path = osp.join(
                        self.data_root, self.data_infos[index]["file_name"]
                    )
                if not osp.exists(img_path):
                    raise ValueError(f"Incorrect image path {img_path}.")

                pil_img = Image.open(img_path).convert("RGB")
                img = np.array(pil_img)
                self.data_infos[index]["img"] = img

                return self.data_infos[index]

    def __getitem__(self, idx):
        if self.sample_rate < 0.999:
            index = self.convert_idx[idx]
            return self.prepare_data(index)
        else:
            return self.prepare_data(idx)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if not self.is_tfrecord:
            for i in range(len(self)):
                img_info = self.getitem_info(i, return_img=False)
                if img_info["width"] / img_info["height"] > 1:
                    self.flag[i] = 1
