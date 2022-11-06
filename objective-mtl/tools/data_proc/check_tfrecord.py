import os
import torch
from io import BytesIO
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from mtl.utils.io_util import imwrite
from mtl.utils.vis_util import imshow
from mtl.utils.tfrecord_util import TFRecordDataSet


class TFRecordImageNetDataSet(TFRecordDataSet):
    def __init__(self, tfrecords, transforms=None):
        super(TFRecordImageNetDataSet, self).__init__(tfrecords)
        self.transforms = transforms

    def parser(self, feature_list):
        label_name = None
        for key, feature in feature_list:
            if key == "name":
                name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            elif key == "image":
                image = feature.bytes_list.value[0]
                if self.transforms is not None:
                    image = Image.open(BytesIO(image))
                    image = image.convert("RGB")
                    image = self.transforms(image)
            elif key == "label":
                label = feature.int64_list.value
            elif key == "label_name":
                label_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
        if label_name is None:
            return name, image, label

        return name, image, label, label_name


def check_imagenet_tfrecord(tfrecord_path_list):
    num_shards = 1
    shard_id = 0
    batch_size = 1
    dataset = TFRecordImageNetDataSet(tfrecord_path_list)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=num_shards, rank=shard_id, shuffle=False
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False
    )

    print("Number of items:%s" % len(dataloader))
    count_num = 0
    for _, data in enumerate(dataloader):
        name, image, label, label_class = data
        print(name, label, label_class)

        image = Image.open(BytesIO(image[0]))
        image = image.convert("RGB")
        np_img = np.array(image)[:, :, [2, 1, 0]]
        imshow(np_img, "dataset_image", 0)
        count_num += 1


if __name__ == "__main__":
    s_path = "data/data_test/test/test_imgs"

    tfr_path_list = ["data/MultiLabelsv2/tfrecords/train.tfrecord"]
    check_imagenet_tfrecord(tfr_path_list, s_path)
