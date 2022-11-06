import os
import struct
import bisect
import tensorflow as tf

from configs import cfg
from mtl.utils.config_util import get_task_cfg
from mtl.utils.config_util import get_dataset_global_args
from mtl.datasets.data_builder import build_dataset
from mtl.utils.gen_tfrecords.gen_tfrecord import bytes_feature, int64_list_feature
from mtl.utils.gen_tfrecords.wy_example_pb2 import Example


def get_info(dataset, index):
    ori_index = dataset.repeat_indices[index]
    dataset_idx = bisect.bisect_right(dataset.dataset.cumulative_sizes, ori_index)
    if dataset_idx == 0:
        sample_idx = ori_index
    else:
        sample_idx = ori_index - dataset.dataset.cumulative_sizes[dataset_idx - 1]

    if dataset.dataset.datasets[dataset_idx].tffiles is None:
        dataset.dataset.datasets[dataset_idx].tffiles = dict()
        for idx, tffile in dataset.dataset.datasets[dataset_idx].idxs:
            dataset.dataset.datasets[dataset_idx].tffiles[tffile] = open(tffile, "rb")

    for idx, tffile in dataset.dataset.datasets[dataset_idx].idxs:
        if sample_idx >= len(idx):
            sample_idx -= len(idx)
            continue

        # every thread keep a f instace
        f = dataset.dataset.datasets[dataset_idx].tffiles[tffile]
        offset = int(idx[sample_idx])

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
        feature_dict = {
            "name": bytes_feature(example.features.feature["name"].bytes_list.value[0]),
            "image": bytes_feature(
                example.features.feature["image"].bytes_list.value[0]
            ),
            "bbox/class": int64_list_feature(
                example.features.feature["bbox/class"].int64_list.value
            ),
            "bbox/xmin": int64_list_feature(
                example.features.feature["bbox/xmin"].int64_list.value
            ),
            "bbox/ymin": int64_list_feature(
                example.features.feature["bbox/ymin"].int64_list.value
            ),
            "bbox/xmax": int64_list_feature(
                example.features.feature["bbox/xmax"].int64_list.value
            ),
            "bbox/ymax": int64_list_feature(
                example.features.feature["bbox/ymax"].int64_list.value
            ),
        }

        return feature_dict


def create_tfr_multiobj_det(dataset, record_path, prefix, split_num):
    """Create the tfrecords

    :param dataset: the object of the dataset class
    :param record_path: the path for saving tfrecord
    :param prefix: the folder of labels
    :param split_num: the number of splits
    """

    single_len = len(dataset) // split_num
    num_tfr_example = 0
    for i in range(split_num):
        tfrecord_path = os.path.join(record_path, prefix + str(i + 1) + ".tfrecord")
        if i == split_num - 1:
            end_idx = len(dataset)
        else:
            end_idx = (i + 1) * single_len

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for idx in range(i * single_len, end_idx):
                feature_dict = get_info(dataset, idx)
                tfr_example = tf.train.Example(
                    features=tf.train.Features(feature=feature_dict)
                )
                writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1
                if num_tfr_example % 100 == 0:
                    print("Create %d TF_Example" % num_tfr_example)


def main_regenerate():

    task_config_path = "tasks/detections/det_yolov4_cspdarknet_multiobj.yaml"
    record_path = "data/objdet-datasets/MultiObjDet/re-tfrecords"
    prefix = "train_"
    # get config
    get_task_cfg(cfg, task_config_path)
    dataset_args = get_dataset_global_args(cfg.DATA)
    train_dataset = build_dataset(
        cfg.DATA.TRAIN_DATA, cfg.DATA.TRAIN_TRANSFORMS, dataset_args
    )

    create_tfr_multiobj_det(train_dataset, record_path, prefix, 15)


if __name__ == "__main__":
    main_regenerate()
