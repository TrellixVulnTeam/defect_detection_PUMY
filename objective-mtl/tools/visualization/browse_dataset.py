import numpy as np
from PIL import Image
from io import BytesIO
import os
import cv2
from torch.utils.data import DataLoader
from PIL import ImageDraw, Image

from mtl.utils.geometric_util import imresize
from mtl.utils.io_util import imwrite
from mtl.utils.tfrecord_util import TFRecordDataSet


class TFRecordImageInfoDataSet(TFRecordDataSet):
    def __init__(self, tfrecords):
        super(TFRecordImageInfoDataSet, self).__init__(tfrecords)

    def parser(self, feature_list):
        for key, feature in feature_list:
            if key == "name" or key == "image_name":
                file_name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            if key == "image":
                image = feature.bytes_list.value[0]
                image = Image.open(BytesIO(image)).convert("RGB")
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif key == "bbox/class":
                label = feature.int64_list.value
            elif key == "label":
                label = feature.int64_list.value

        return file_name, img, label


def check_obj_tfrecord(tfrecord_path_list, save_path):
    dataset = TFRecordImageInfoDataSet(tfrecord_path_list)
    sampler = None
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=4, pin_memory=False
    )

    print("Number of items:%s" % len(dataloader))
    count_num = 0
    for _, data in enumerate(dataloader):
        name, image, label = data
        print(name, label)

        # imshow(np_img, 'dataset_image', 0)
        imwrite(
            image.detach().cpu().numpy()[0], os.path.join(save_path, name[0] + ".jpg")
        )
        count_num += 1

        if count_num == 10:
            break


def get_label_list(data_path, label_map):
    label_info_list = open(os.path.join(data_path, label_map)).readlines()

    label_list = []
    for x in label_info_list[1:]:
        tmp_splits = x.split("\t")
        if len(tmp_splits) >= 4:
            label_list.append(tmp_splits[3].strip())
    return label_list


def show_tencentml_annotations(data_path, tfr_path_list, label_list, save_dir):
    tfrecord_path_list = [
        os.path.join(data_path, tfr_path) for tfr_path in tfr_path_list
    ]
    dataset = TFRecordImageInfoDataSet(tfrecord_path_list)
    sampler = None
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=4, pin_memory=False
    )

    print("Number of items:%s" % len(dataloader))
    for _, data in enumerate(dataloader):
        img_name, img, label_vector = data
        img = img[0].numpy()
        img_name = img_name[0]
        img_h = img.shape[0]
        img_w = img.shape[1]
        re_w = 512
        re_h = int((img_h / img_w) * re_w)
        img = imresize(img, [re_w, re_h])

        img = cv2.copyMakeBorder(img, 0, 0, 0, 200, cv2.BORDER_CONSTANT)
        cv2.destroyAllWindows()

        # draw character
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        pos = (520, 0)
        text = "Labels:"
        draw.text(pos, text, fill=(255, 255, 255))

        pos_2 = 20
        for label_index in label_vector:
            pos = (520, pos_2)
            draw.text(pos, label_list[label_index.numpy()[0]], fill=(255, 255, 255))
            pos_2 += 20

        img_ocv = np.array(img_pil)  # PIL to numpy
        img = cv2.cvtColor(img_ocv, cv2.COLOR_RGB2BGR)  # PIL to OpenCV bgr

        # imshow(img, win_name=img_name.split('/')[1], wait_time=0)
        imwrite(
            img,
            os.path.join(
                save_dir, img_name.split("/")[0] + "_" + img_name.split("/")[1]
            ),
        )


if __name__ == "__main__":
    # tfr_path_list = ["data/objdet-datasets/MultiObjDet/ori-tfrecords/train_04.tfrecord"]
    # save_dir = "meta/test_res"
    # check_obj_tfrecord(tfr_path_list, save_dir)

    # show for the TencentML datasets
    DATA_PATH = "data/objcls-datasets/TencentML"
    LABEL_MAP_FILE = "meta/dictionary_and_semantic_hierarchy.txt"
    TFR_PATH_LIST = ["tfrecords/imagenet_train_53.tfrecord"]
    SAVE_DIR = "data/objcls-datasets/TencentML/images"

    ind_label_list = get_label_list(DATA_PATH, LABEL_MAP_FILE)
    show_tencentml_annotations(DATA_PATH, TFR_PATH_LIST, ind_label_list, SAVE_DIR)
