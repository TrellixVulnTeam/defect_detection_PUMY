# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 22:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn
# @File    : gen_tfrecord.py

import json
import tensorflow as tf
import os
from PIL import Image
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import numpy as np

from mtl.utils.io_util import file_load


mtl_class_names = (
    "person",
    "cartoon-person",
    "game-role",
    "cat",
    "dog",
    "snake",
    "bird",
    "fish",
    "rabbit",
    "monkey",
    "horse",
    "chicken",
    "pig",
    "cow",
    "sheep",
    "bicycle",
    "tricycle",
    "motorbike",
    "tractor",
    "car",
    "bus",
    "truck",
    "excavator",
    "crane",
    "train",
    "plane",
    "tank",
    "ship",
    "villa",
    "pavilion",
    "tower",
    "temple",
    "palace",
    "chair",
    "bed",
    "table",
    "sofa",
    "bench",
    "vase",
    "potted-plant",
    "bag",
    "umbrella",
    "computer",
    "television",
    "lamp",
    "mouse",
    "keyboard",
    "cell-phone",
    "dish",
    "bowl",
    "spoon",
    "bottle",
    "cup",
    "fork",
    "pot",
    "knife",
    "basketball",
    "skateboard",
    "book",
    "banana",
    "apple",
    "orange",
    "watermelon",
    "pizza",
    "cake",
)
mtl_cat2label = {cat: i for i, cat in enumerate(mtl_class_names)}

coco_class_names = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

voc_class_names = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
voc_cat2label = {cat: i for i, cat in enumerate(voc_class_names)}


def float_feature(value):
    """The float for describing features

    :param value: the input float
    :return: the feature in tfrecord format
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """The float list for describing features

    :param value: the input float list
    :return: the feature in tfrecord format
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    """The int value for describing features

    :param value: the input int
    :return: the feature in tfrecord format
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """The int list for describing features

    :param value: the input int list
    :return: the feature in tfrecord format
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """The bytes list for describing feature

    :param value: the input bytes list
    :return: the feature in tfrecord format
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """
    The bytes list for describing feature
    :param value: the input bytes list
    :return: the feature in tfrecord format
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def read_file_list(list_path):
    """
    Read the dataset information file by lines
    :param list_path: the path of the dataset info file
    :return: the list of data information
    """
    with open(list_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() != ""]


def read_bbox_list(label_path, split_str=" "):
    """Read the object list from the label path

    :param label_path: the path of the label file
    :return: the object list
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    bbox_class = []
    bbox_xmin = []
    bbox_ymin = []
    bbox_xmax = []
    bbox_ymax = []
    for line in lines:
        obj_raw = line.strip()
        if obj_raw == "":
            continue
        obj_info = obj_raw.split(split_str)
        if len(obj_info) < 5:
            if len(obj_info) > 0:
                print(obj_info)
            continue

        bbox_class.append(int(obj_info[0]))
        bbox_xmin.append(int(obj_info[1]))
        bbox_ymin.append(int(obj_info[2]))
        bbox_xmax.append(int(obj_info[3]))
        bbox_ymax.append(int(obj_info[4]))
        # obj_bboxes = [int(obj_info[i]) for i in range(5)]
    return bbox_class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax


def read_voc_bbox_list(label_path, width, height):
    tree = ET.parse(label_path)
    root = tree.getroot()
    bbox_class = []
    bbox_xmin = []
    bbox_ymin = []
    bbox_xmax = []
    bbox_ymax = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in voc_class_names:
            print("Name error: ", label_path)
            continue

        label = voc_cat2label[name]

        bnd_box = obj.find("bndbox")

        bbox = [
            int(float(bnd_box.find("xmin").text)),
            int(float(bnd_box.find("ymin").text)),
            int(float(bnd_box.find("xmax").text)),
            int(float(bnd_box.find("ymax").text)),
        ]

        if bbox[0] > bbox[2]:
            tmp = bbox[0]
            bbox[0] = bbox[2]
            bbox[2] = tmp
        if bbox[1] > bbox[3]:
            tmp = bbox[1]
            bbox[1] = bbox[3]
            bbox[3] = tmp

        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[0] >= width:
            bbox[0] = width - 1
        if bbox[2] < 0:
            bbox[2] = 0
        if bbox[2] >= width:
            bbox[2] = width - 1
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[1] >= height:
            bbox[1] = height - 1
        if bbox[3] < 0:
            bbox[3] = 0
        if bbox[3] >= height:
            bbox[3] = height - 1

        if bbox[2] < bbox[0] + 5 and bbox[3] < bbox[1] + 5:
            print("Ignored bbox: ", label_path, bbox)
            continue

        bbox_class.append(label)
        bbox_xmin.append(bbox[0])
        bbox_ymin.append(bbox[1])
        bbox_xmax.append(bbox[2])
        bbox_ymax.append(bbox[3])

    return bbox_class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax


def read_yolo_bbox_list(label_path, width, height, split_str=" "):
    """Read the object list from the label path

    :param label_path: the path of the label file
    :return: the object list
    """
    with open(label_path, "r") as fr:
        lines = fr.readlines()
    bbox_class = []
    bbox_xmin = []
    bbox_ymin = []
    bbox_xmax = []
    bbox_ymax = []
    for line in lines:
        obj_raw = line.strip()
        if obj_raw == "":
            continue
        obj_info = obj_raw.split(split_str)
        if len(obj_info) < 5:
            if len(obj_info) > 0:
                print(obj_info)
            continue

        bbox_cx = float(obj_info[1]) * width
        bbox_cy = float(obj_info[2]) * height
        bbox_w = float(obj_info[3]) * width
        bbox_h = float(obj_info[4]) * height

        bbox_class.append(int(obj_info[0]))
        bbox_xmin.append(int(bbox_cx - bbox_w / 2 + 0.5))
        bbox_ymin.append(int(bbox_cy - bbox_h / 2 + 0.5))
        bbox_xmax.append(int(bbox_cx + bbox_w / 2 + 0.5))
        bbox_ymax.append(int(bbox_cy + bbox_h / 2 + 0.5))

        # obj_bboxes = [int(obj_info[i]) for i in range(5)]
    return bbox_class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax


def create_tfr_det_example(
    dataset_path, img_dir, label_dir, file_name, label_format, split_str=" "
):
    """Create the instance with tfrecord format for the file

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param label_dir: the folder of labels
    :param file_name: the name of image or label file
    """
    img_path = os.path.join(dataset_path, img_dir, file_name + ".jpg")
    if not os.path.isfile(img_path):
        img_path = os.path.join(dataset_path, img_dir, file_name + ".png")
        if not os.path.isfile(img_path):
            return None

    if label_format == "voc":
        label_path = os.path.join(dataset_path, label_dir, file_name + ".xml")
    else:
        label_path = os.path.join(dataset_path, label_dir, file_name + ".txt")
    if not os.path.isfile(label_path):
        return None

    image = open(img_path, "rb").read()  # read image to memory as Bytes

    try:
        # print(img_path)
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        print(img_path)
        return None
    width, height = pil_img.size

    if label_format == "yolo":
        bbox_class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = read_yolo_bbox_list(
            label_path, width, height, split_str=split_str
        )
    elif label_format == "voc":
        bbox_class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = read_voc_bbox_list(
            label_path, width, height
        )
    else:
        bbox_class, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = read_bbox_list(
            label_path, split_str=split_str
        )

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(file_name.encode()),
                "image": bytes_feature(image),
                "bbox/class": int64_list_feature(bbox_class),
                "bbox/xmin": int64_list_feature(bbox_xmin),
                "bbox/ymin": int64_list_feature(bbox_ymin),
                "bbox/xmax": int64_list_feature(bbox_xmax),
                "bbox/ymax": int64_list_feature(bbox_ymax),
            }
        )
    )
    return tfr_example


def create_tfr_obj_example(file_name, img_path, ann_dict):
    image = open(img_path, "rb").read()  # read image to memory as Bytes
    try:  # for test the intergrity
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        print(img_path)
        return None

    bbox_class = []
    bbox_xmin = []
    bbox_ymin = []
    bbox_xmax = []
    bbox_ymax = []
    # print(len(ann_dict["labels"]))
    for i in range(len(ann_dict["labels"])):
        bbox_class.append(int(ann_dict["labels"][i]))
        bbox_xmin.append(int(ann_dict["bboxes"][i][0] + 0.5))
        bbox_ymin.append(int(ann_dict["bboxes"][i][1] + 0.5))
        bbox_xmax.append(int(ann_dict["bboxes"][i][2] + 0.5))
        bbox_ymax.append(int(ann_dict["bboxes"][i][3] + 0.5))

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(file_name.encode()),
                "image": bytes_feature(image),
                "bbox/class": int64_list_feature(bbox_class),
                "bbox/xmin": int64_list_feature(bbox_xmin),
                "bbox/ymin": int64_list_feature(bbox_ymin),
                "bbox/xmax": int64_list_feature(bbox_xmax),
                "bbox/ymax": int64_list_feature(bbox_ymax),
            }
        )
    )
    return tfr_example


def create_tfr_cls_example(img_path, file_id, label, height, width):
    """Create the instance with tfrecord format for the file

    :param img_path (str): the path of the image
    :param label (int): the label of the image
    :param height (int): the height of the image
    :param width (int): the width of the image
    """
    if not os.path.isfile(img_path):
        return None

    image = open(img_path, "rb").read()  # read image to memory as Bytes

    if len(image) == 0:
        return None

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(file_id.encode()),
                "image": bytes_feature(image),
                "label": int64_feature(label),
                "height": int64_feature(height),
                "width": int64_feature(width),
            }
        )
    )
    return tfr_example


def create_tfr_seg_example(dataset_path, img_dir, seg_dir, file_name):
    """Create the instance with tfrecord format for the file

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param seg_dir: the folder of segmentations
    :param file_name: the name of image or seg file
    """
    img_path = os.path.join(dataset_path, img_dir, file_name + ".jpg")
    if not os.path.isfile(img_path):
        return None

    seg_path = os.path.join(dataset_path, seg_dir, file_name + ".png")
    if not os.path.isfile(seg_path):
        return None

    image = open(img_path, "rb").read()  # read image to memory as Bytes

    if len(image) == 0:
        return None
    # try:
    #     pil_img = Image.open(img_path).convert("RGB")
    # except Exception:
    #     print(img_path)
    #     return None

    seg_image = open(seg_path, "rb").read()  # read image to memory as Bytes

    if len(seg_image) == 0:
        return None
    # try:
    #     seg_pil_img = Image.open(seg_path).convert("RGB")
    # except Exception:
    #     print(seg_path)
    #     return None

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(file_name.encode()),
                "image": bytes_feature(image),
                "segmentation": bytes_feature(seg_image),
            }
        )
    )
    return tfr_example


def read_mtl_bbox_list(label_path, width, height):
    key_node_name = [
        "headtop",
        "nose",
        "lefteye",
        "righteye",
        "leftear",
        "rightear",
        "leftshoulder",
        "rightshoulder",
        "leftelbow",
        "rightelbow",
        "leftwrist",
        "rightwrist",
        "lefthip",
        "righthip",
        "leftknee",
        "rightknee",
        "leftankle",
        "rightankle",
    ]

    tree = ET.parse(label_path)
    root = tree.getroot()
    bbox_info = [[], [], [], [], [], [], []]
    has_property = []
    person_property = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in mtl_class_names:
            print("Name error: ", label_path)
            continue
        try:
            truncated = int(obj.find("truncated").text)
        except AttributeError:
            truncated = 0
        try:
            difficult = int(obj.find("difficult").text)
        except AttributeError:
            difficult = 0

        label = mtl_cat2label[name]

        try:
            bnd_box = obj.find("bndbox")
            bbox = [
                int(float(bnd_box.find("xmin").text)),
                int(float(bnd_box.find("ymin").text)),
                int(float(bnd_box.find("xmax").text)),
                int(float(bnd_box.find("ymax").text)),
            ]
        except AttributeError:
            continue

        if bbox[0] > bbox[2]:
            tmp = bbox[0]
            bbox[0] = bbox[2]
            bbox[2] = tmp
        if bbox[1] > bbox[3]:
            tmp = bbox[1]
            bbox[1] = bbox[3]
            bbox[3] = tmp

        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[0] >= width:
            bbox[0] = width - 1
        if bbox[2] < 0:
            bbox[2] = 0
        if bbox[2] >= width:
            bbox[2] = width - 1
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[1] >= height:
            bbox[1] = height - 1
        if bbox[3] < 0:
            bbox[3] = 0
        if bbox[3] >= height:
            bbox[3] = height - 1

        if bbox[2] < bbox[0] + 5 or bbox[3] < bbox[1] + 5:
            print("Ignored bbox: ", label_path, bbox)
            continue

        bbox_info[0].append(label)
        bbox_info[1].append(bbox[0])
        bbox_info[2].append(bbox[1])
        bbox_info[3].append(bbox[2])
        bbox_info[4].append(bbox[3])
        bbox_info[5].append(truncated)
        bbox_info[6].append(difficult)

        if name == "person":
            gender_handle = obj.find("gender")
            age_handle = obj.find("age")
            key_node_handle = obj.find("keynode")
            if (
                (gender_handle is not None)
                or (age_handle is not None)
                or (key_node_handle is not None)
            ):
                # the node may be lost
                has_property.append(1)
                key_nodes = []
                for i in range(len(key_node_name)):
                    if key_node_handle is not None:
                        if key_node_handle.find(key_node_name[i]) is not None:
                            key_nodes.append(
                                eval(key_node_handle.find(key_node_name[i]).text)
                            )
                        else:
                            key_nodes.append((-1, -1))
                    else:
                        key_nodes.append((-1, -1))

                if gender_handle is not None:
                    if gender_handle.text == "female":
                        gender = 1
                    else:
                        gender = 0
                else:
                    gender = 0

                if age_handle is not None:
                    if age_handle.text == "child":
                        age = 1
                    else:
                        age = 0
                else:
                    age = 0

                person_property[0].append(gender)
                person_property[1].append(age)
                for i in range(len(key_nodes)):
                    person_property[2 * i + 2].append(int(key_nodes[i][0] + 0.5))
                    person_property[2 * i + 3].append(int(key_nodes[i][1] + 0.5))
            else:
                has_property.append(0)
        else:
            has_property.append(0)

    return bbox_info, has_property, person_property


def create_tfr_mtl_example(dataset_path, img_dir, label_dir, file_name):
    """Create the instance with tfrecord format for the file

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param label_dir: the folder of labels
    :param file_name: the name of image or label file
    """
    img_path = os.path.join(dataset_path, img_dir, file_name + ".jpg")
    if not os.path.isfile(img_path):
        img_path = os.path.join(dataset_path, img_dir, file_name + ".png")
        if not os.path.isfile(img_path):
            return None

    label_path = os.path.join(dataset_path, label_dir, file_name + ".xml")

    if not os.path.isfile(label_path):
        return None

    image = open(img_path, "rb").read()  # read image to memory as Bytes
    try:
        # print(img_path)
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        print(img_path)
        return None
    width, height = pil_img.size

    bbox_info, has_property, person_property = read_mtl_bbox_list(
        label_path, width, height
    )

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(file_name.encode()),
                "image": bytes_feature(image),
                "bbox/class": int64_list_feature(bbox_info[0]),
                "bbox/xmin": int64_list_feature(bbox_info[1]),
                "bbox/ymin": int64_list_feature(bbox_info[2]),
                "bbox/xmax": int64_list_feature(bbox_info[3]),
                "bbox/ymax": int64_list_feature(bbox_info[4]),
                "truncated": int64_list_feature(bbox_info[5]),
                "difficult": int64_list_feature(bbox_info[6]),
                "property": int64_list_feature(has_property),
                "gender": int64_list_feature(person_property[0]),
                "age": int64_list_feature(person_property[1]),
                "headtop/x": int64_list_feature(person_property[2]),
                "headtop/y": int64_list_feature(person_property[3]),
                "nose/x": int64_list_feature(person_property[4]),
                "nose/y": int64_list_feature(person_property[5]),
                "lefteye/x": int64_list_feature(person_property[6]),
                "lefteye/y": int64_list_feature(person_property[7]),
                "righteye/x": int64_list_feature(person_property[8]),
                "righteye/y": int64_list_feature(person_property[9]),
                "leftear/x": int64_list_feature(person_property[10]),
                "leftear/y": int64_list_feature(person_property[11]),
                "rightear/x": int64_list_feature(person_property[12]),
                "rightear/y": int64_list_feature(person_property[13]),
                "leftshoulder/x": int64_list_feature(person_property[14]),
                "leftshoulder/y": int64_list_feature(person_property[15]),
                "rightshoulder/x": int64_list_feature(person_property[16]),
                "rightshoulder/y": int64_list_feature(person_property[17]),
                "leftelbow/x": int64_list_feature(person_property[18]),
                "leftelbow/y": int64_list_feature(person_property[19]),
                "rightelbow/x": int64_list_feature(person_property[20]),
                "rightelbow/y": int64_list_feature(person_property[21]),
                "leftwrist/x": int64_list_feature(person_property[22]),
                "leftwrist/y": int64_list_feature(person_property[23]),
                "rightwrist/x": int64_list_feature(person_property[24]),
                "rightwrist/y": int64_list_feature(person_property[25]),
                "lefthip/x": int64_list_feature(person_property[26]),
                "lefthip/y": int64_list_feature(person_property[27]),
                "righthip/x": int64_list_feature(person_property[28]),
                "righthip/y": int64_list_feature(person_property[29]),
                "leftknee/x": int64_list_feature(person_property[30]),
                "leftknee/y": int64_list_feature(person_property[31]),
                "rightknee/x": int64_list_feature(person_property[32]),
                "rightknee/y": int64_list_feature(person_property[33]),
                "leftankle/x": int64_list_feature(person_property[34]),
                "leftankle/y": int64_list_feature(person_property[35]),
                "rightankle/x": int64_list_feature(person_property[36]),
                "rightankle/y": int64_list_feature(person_property[37]),
            }
        )
    )
    return tfr_example


def create_tfr_headnode_example(img_path, anno_path):
    """Create the instance with tfrecord format for the file

    :param img_path (str): the path of the image
    :param label (int): the label of the image
    :param height (int): the height of the image
    :param width (int): the width of the image
    """
    if not os.path.isfile(img_path):
        return None
    if not os.path.isfile(anno_path):
        return None

    image = open(img_path, "rb").read()  # read image to memory as Bytes
    pil_img = Image.open(img_path).convert("RGB")
    width, height = pil_img.size

    anno_data = file_load(anno_path)
    left_eye_pt = []
    right_eye_pt = []
    face_box = []

    for obj in anno_data["regions"]:
        if obj["tags"][0] == "left eye":
            left_eye_pt.append([obj["boundingBox"]["left"], obj["boundingBox"]["top"]])
        elif obj["tags"][0] == "right eye":
            right_eye_pt.append([obj["boundingBox"]["left"], obj["boundingBox"]["top"]])
        elif obj["tags"][0] == "face":
            face_box.append(
                [
                    obj["boundingBox"]["left"],
                    obj["boundingBox"]["top"],
                    obj["boundingBox"]["left"] + obj["boundingBox"]["width"],
                    obj["boundingBox"]["top"] + obj["boundingBox"]["height"],
                ]
            )
    bbox_class = []
    bbox_xmin = []
    bbox_ymin = []
    bbox_xmax = []
    bbox_ymax = []
    xkeynode = []
    ykeynode = []
    for i in range(len(face_box)):
        bbox_class.append(0)
        bbox_xmin.append(int(face_box[i][0] + 0.5))
        bbox_ymin.append(int(face_box[i][1] + 0.5))
        bbox_xmax.append(int(face_box[i][2] + 0.5))
        bbox_ymax.append(int(face_box[i][3] + 0.5))
        x_node = 0
        y_node = 0
        count = 0
        for left_eye in left_eye_pt:
            if (
                (left_eye[0] >= face_box[i][0])
                and (left_eye[0] <= face_box[i][2])
                and (left_eye[1] >= face_box[i][1])
                and (left_eye[1] <= face_box[i][3])
            ):
                x_node += int(left_eye[0])
                y_node += int(left_eye[1])
                count += 1
        for right_eye in right_eye_pt:
            if (
                (right_eye[0] >= face_box[i][0])
                and (right_eye[0] <= face_box[i][2])
                and (right_eye[1] >= face_box[i][1])
                and (right_eye[1] <= face_box[i][3])
            ):
                x_node += int(right_eye[0])
                y_node += int(right_eye[1])
                count += 1
        if count < 1:
            x_node = int((face_box[i][0] + face_box[i][2]) / 2 + 0.5)
            y_node = int((face_box[i][1] + face_box[i][3]) / 2 + 0.5)
        else:
            x_node = int(x_node / count + 0.5)
            y_node = int(y_node / count + 0.5)
        xkeynode.append(x_node)
        ykeynode.append(y_node)

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image_name": bytes_feature(
                    img_path.split("/")[-1].split(".")[0].encode()
                ),
                "image": bytes_feature(image),
                "height": int64_feature(height),
                "width": int64_feature(width),
                "bbox/class": int64_list_feature(bbox_class),
                "bbox/xmin": int64_list_feature(bbox_xmin),
                "bbox/ymin": int64_list_feature(bbox_ymin),
                "bbox/xmax": int64_list_feature(bbox_xmax),
                "bbox/ymax": int64_list_feature(bbox_ymax),
                "bbox/xkeynode": int64_list_feature(xkeynode),
                "bbox/ykeynode": int64_list_feature(ykeynode),
            }
        )
    )
    return tfr_example


def generate_det_tfrecord(
    dataset_path,
    img_dir,
    label_dir,
    list_store_dir,
    list_name,
    record_path,
    label_format="normal",
    split_str=" ",
):
    """Generate the tfrecord for the content of file list in the dataset

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param label_dir: the folder of labels
    :param list_store_dir: the folder for storing splitted file lists
    :param list_name: the name of list file
    :param tfrecord_path: the path for storing generated tfrecord
    :return: none
    """
    tfrecord_path = os.path.join(
        dataset_path, record_path, list_name[:-4] + ".tfrecord"
    )
    list_path = os.path.join(dataset_path, list_store_dir, list_name)
    list_files = read_file_list(list_path)

    num_tfr_example = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for file_name in list_files:
            # if num_tfr_example >= 16800 and num_tfr_example < 16900:
            #     print(file_name)
            tfr_example = create_tfr_det_example(
                dataset_path, img_dir, label_dir, file_name, label_format, split_str
            )
            if tfr_example is not None:
                writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1
                if num_tfr_example % 100 == 0:
                    print("Create %d TF_Example" % num_tfr_example)

    print(
        "{} examples has been created, which are saved in {}".format(
            num_tfr_example, tfrecord_path
        )
    )


def generate_cls_tfrecord(
    dataset_path, img_dir, list_store_dir, list_name, record_path, is_file_ext=False
):
    """Generate the tfrecord for the content of file list in the dataset

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param label_dir: the folder of labels
    :param list_store_dir: the folder for storing splitted file lists
    :param list_name: the name of list file
    :param tfrecord_path: the path for storing generated tfrecord
    :return: none
    """
    tfrecord_path = os.path.join(
        dataset_path, record_path, list_name[:-4] + ".tfrecord"
    )

    list_path = os.path.join(dataset_path, list_store_dir, list_name)
    list_files = read_file_list(list_path)

    num_tfr_example = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for file_info in list_files:
            str_list = file_info.split()
            if len(str_list) >= 4:
                file_id = str_list[0]
                label = int(str_list[1])
                height = int(str_list[2])
                width = int(str_list[3])
                if is_file_ext:
                    img_path = os.path.join(dataset_path, img_dir, file_id)
                else:
                    img_path = os.path.join(dataset_path, img_dir, file_id + ".jpg")

                tfr_example = create_tfr_cls_example(
                    img_path, file_id, label, height, width
                )
                if tfr_example is not None:
                    writer.write(tfr_example.SerializeToString())
                    num_tfr_example += 1
                    if num_tfr_example % 100 == 0:
                        print("Create %d TF_Example" % num_tfr_example)
                else:
                    print("Error in loading: ", img_path)

    print(
        "{} examples has been created, which are saved in {}".format(
            num_tfr_example, tfrecord_path
        )
    )


def generate_seg_tfrecord(
    dataset_path, img_dir, seg_dir, list_store_dir, list_name, record_path
):
    """Generate the tfrecord for the content of file list in the dataset

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param seg_dir: the folder of segmentations
    :param list_store_dir: the folder for storing splitted file lists
    :param list_name: the name of list file
    :param tfrecord_path: the path for storing generated tfrecord
    :return: none
    """
    tfrecord_path = os.path.join(
        dataset_path, record_path, list_name[:-4] + ".tfrecord"
    )
    list_path = os.path.join(dataset_path, list_store_dir, list_name)
    list_files = read_file_list(list_path)

    num_tfr_example = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for file_name in list_files:
            tfr_example = create_tfr_seg_example(
                dataset_path, img_dir, seg_dir, file_name
            )
            if tfr_example is not None:
                writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1
                if num_tfr_example % 100 == 0:
                    print("Create %d TF_Example" % num_tfr_example)

    print(
        "{} examples has been created, which are saved in {}".format(
            num_tfr_example, tfrecord_path
        )
    )


def generate_mtl_tfrecord(
    dataset_path, img_dir, label_dir, list_store_dir, list_name, record_path
):
    """Generate the tfrecord for the content of file list in the dataset

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param label_dir: the folder of labels
    :param list_store_dir: the folder for storing splitted file lists
    :param list_name: the name of list file
    :param tfrecord_path: the path for storing generated tfrecord
    :return: none
    """
    tfrecord_path = os.path.join(
        dataset_path, record_path, list_name[:-4] + ".tfrecord"
    )
    list_path = os.path.join(dataset_path, list_store_dir, list_name)
    list_files = read_file_list(list_path)

    num_tfr_example = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for file_name in list_files:
            tfr_example = create_tfr_mtl_example(
                dataset_path, img_dir, label_dir, file_name
            )
            if tfr_example is not None:
                writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1
                if num_tfr_example % 100 == 0:
                    print("Create %d TF_Example" % num_tfr_example)

    print(
        "{} examples has been created, which are saved in {}".format(
            num_tfr_example, tfrecord_path
        )
    )


def get_coco_data_infos(ann_file):
    coco_dataset = COCO(ann_file)
    coco_cat_ids = coco_dataset.getCatIds(catNms=coco_class_names)
    coco_cat2label = {cat_id: i for i, cat_id in enumerate(coco_cat_ids)}
    coco_img_ids = coco_dataset.getImgIds()
    data_infos = []
    for i in coco_img_ids:
        info = coco_dataset.loadImgs([i])[0]
        data_infos.append(info)
    return data_infos, coco_dataset, coco_cat2label, coco_cat_ids


def parse_coco_ann_info(img_info, ann_info, coco_cat2label, coco_cat_ids):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []
    for _, ann in enumerate(ann_info):
        if ann.get("ignore", False):
            continue
        x1, y1, w, h = ann["bbox"]
        inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
        if inter_w * inter_h == 0:
            continue
        if ann["area"] <= 0 or w < 1 or h < 1:
            continue
        if ann["category_id"] not in coco_cat_ids:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get("iscrowd", False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_labels.append(coco_cat2label[ann["category_id"]])
            gt_masks_ann.append(ann.get("segmentation", None))

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    seg_map = img_info["file_name"].replace("jpg", "png")

    ann = dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        bboxes_ignore=gt_bboxes_ignore,
        masks=gt_masks_ann,
        seg_map=seg_map,
    )

    return ann


def generate_coco_tfrecord(
    dataset_path, train_img_dir, train_anno_path, record_path, record_name
):
    train_ann_file = os.path.join(dataset_path, train_anno_path)
    train_data_infos, train_dataset, coco_cat2label, coco_cat_ids = get_coco_data_infos(
        train_ann_file
    )
    train_tfrecord_path = os.path.join(
        dataset_path, record_path, record_name + ".tfrecord"
    )
    num_tfr_example = 0
    with tf.io.TFRecordWriter(train_tfrecord_path) as writer:
        for file_info in train_data_infos:
            img_ind = file_info["id"]
            ann_ids = train_dataset.getAnnIds(imgIds=[img_ind])
            ann_info = train_dataset.loadAnns(ann_ids)
            ann_dict = parse_coco_ann_info(
                file_info, ann_info, coco_cat2label, coco_cat_ids
            )
            img_path = os.path.join(dataset_path, train_img_dir, file_info["file_name"])
            if not os.path.exists(img_path):
                raise ValueError(f"Incorrect image path {img_path}.")
            tfr_example = create_tfr_obj_example(
                file_info["file_name"], img_path, ann_dict
            )
            if tfr_example is not None:
                writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1
                if num_tfr_example % 100 == 0:
                    print("Create %d TF_Example" % num_tfr_example)
            else:
                print("Error in loading: ", img_path)
        print(
            "{} examples has been created, which are saved in {}".format(
                num_tfr_example, train_tfrecord_path
            )
        )


def generate_object365_tfrecord(
    dataset_path, img_dir, anno_path, record_path, record_name
):
    ann_file = os.path.join(dataset_path, anno_path)
    with open(ann_file, "r") as f:
        ann_data = json.load(f)
        print("anno count:", len(ann_data["annotations"]))
        print("image count:", len(ann_data["images"]))

        img_map = {}
        img_anno_infos = {}
        for item in ann_data["images"]:
            img_map[item["id"]] = item
            img_anno_infos[item["id"]] = []

        for anno in ann_data["annotations"]:
            img_anno_infos[anno["image_id"]].append(anno)

    num_tfr_example = 0
    for _, img_id in enumerate(img_anno_infos):
        num_idx = num_tfr_example // 100000
        tfrecord_path = os.path.join(
            dataset_path, record_path, record_name + "_" + str(num_idx) + ".tfrecord"
        )
        if num_tfr_example % 100000 == 0:
            writer = tf.io.TFRecordWriter(tfrecord_path)

        annos = img_anno_infos[img_id]
        img = img_map[img_id]
        file_parts = img["file_name"].split("/")
        img_file_name = os.path.join(file_parts[-2], file_parts[-1])
        img_path = os.path.join(dataset_path, img_dir, img_file_name)
        if not os.path.exists(img_path):
            print("Error path:", img_path)
            continue
        ann_dict = {}
        img_width = img["width"]
        img_height = img["height"]
        labels = []
        bboxes = []
        for anno in annos:
            bbox = anno["bbox"]
            xmin = max(int(bbox[0]), 0)
            ymin = max(int(bbox[1]), 0)
            xmax = min(int(bbox[2] + bbox[0]), img_width)
            ymax = min(int(bbox[3] + bbox[1]), img_height)
            labels.append(anno["category_id"])
            bboxes.append([xmin, ymin, xmax, ymax])
        ann_dict["labels"] = labels
        ann_dict["bboxes"] = bboxes
        tfr_example = create_tfr_obj_example(img_file_name, img_path, ann_dict)
        if tfr_example is not None:
            writer.write(tfr_example.SerializeToString())
            num_tfr_example += 1
            if num_tfr_example % 100 == 0:
                print("Create %d TF_Example" % num_tfr_example)
        else:
            print("Error in loading: ", img_path)

    print("{} examples has been created.".format(num_tfr_example))


def generate_lupinus_tfrecord(
    dataset_path,
    data_dir,
    record_dir="tfrecords",
    class_names_file="class-names-list.txt",
    store_file=True,
):
    data_path = os.path.join(dataset_path, data_dir)
    if store_file:
        class_names_list = []
        for sub_dir in os.listdir(data_path):
            if sub_dir.startswith("."):
                continue
            sub_path = os.path.join(data_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            for sub_file in os.listdir(sub_path):
                if sub_file.endswith(".json"):
                    with open(os.path.join(sub_path, sub_file), "r") as jf:
                        ann_data = json.load(jf)
                        for ann_bbox in ann_data["shapes"]:
                            if ann_bbox["label"] not in class_names_list:
                                class_names_list.append(ann_bbox["label"])
        class_file_path = os.path.join(dataset_path, class_names_file)
        fw = open(class_file_path, "w")
        for cls_name in class_names_list:
            fw.write(cls_name + "\n")
        fw.close()
    else:
        class_file_path = os.path.join(dataset_path, class_names_file)
        fr = open(class_file_path, "r")
        class_names_list = []
        for cls_name in fr.readlines():
            cls_name = cls_name.strip()
            if len(cls_name) > 0:
                class_names_list.append(cls_name)
        fr.close()

    lupinus_cat2label = {cat: i for i, cat in enumerate(class_names_list)}
    # find image and annotation pairs
    for sub_dir in os.listdir(data_path):
        if sub_dir.startswith("."):
            continue
        sub_path = os.path.join(data_path, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        if sub_dir.startswith("train") or sub_dir.startswith("val"):
            tfrecord_path = os.path.join(
                dataset_path, record_dir, sub_dir + ".tfrecord"
            )
        else:
            tfrecord_path = os.path.join(
                dataset_path, record_dir, "train_" + sub_dir + ".tfrecord"
            )

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            num_tfr_example = 0
            for sub_file in os.listdir(sub_path):
                if sub_file.endswith(".json"):
                    anno_path = os.path.join(sub_path, sub_file)
                    img_file = sub_file[:-4] + "jpg"
                    img_path = os.path.join(sub_path, img_file)
                    if not os.path.isfile(img_path):
                        print("unpaired json file: ", anno_path)
                        continue
                    ann_dict = {}
                    with open(anno_path, "r") as jf:
                        ann_data = json.load(jf)
                    img_width = ann_data["imageWidth"]
                    img_height = ann_data["imageHeight"]

                    labels = []
                    bboxes = []
                    for anno in ann_data["shapes"]:
                        xmin = img_width
                        ymin = img_height
                        xmax = 0
                        ymax = 0
                        for pt in anno["points"]:
                            if pt[0] < xmin:
                                xmin = max(int(pt[0]), 0)
                            if pt[1] < ymin:
                                ymin = max(int(pt[1]), 0)
                            if pt[0] > xmax:
                                xmax = min(int(pt[0]), img_width)
                            if pt[1] > ymax:
                                ymax = min(int(pt[1]), img_height)

                        labels.append(lupinus_cat2label[anno["label"]])
                        bboxes.append([xmin, ymin, xmax, ymax])
                    ann_dict["labels"] = labels
                    ann_dict["bboxes"] = bboxes
                    tfr_example = create_tfr_obj_example(img_file, img_path, ann_dict)
                    if tfr_example is not None:
                        writer.write(tfr_example.SerializeToString())
                        num_tfr_example += 1
                        if num_tfr_example % 100 == 0:
                            print("Create %d TF_Example" % num_tfr_example)
                    else:
                        print("Error in loading: ", img_path)

            print(
                "{} examples has been created for {}.".format(
                    num_tfr_example, tfrecord_path
                )
            )
