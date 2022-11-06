import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import traceback
from PIL import Image
import shutil

from mtl.utils.io_util import imwrite, file_load
from mtl.utils.vis_util import imshow_det_bboxes


class_names = [
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
]

cat2label = {cat: i for i, cat in enumerate(class_names)}

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

# key_node_name = [
#     'headtop', 'nose', 'lefteye', 'righteye', 'leftear', 'rightear',
#     'shoulderleft', 'shoulderright','elbowleft', 'elbowright', 'wristleft',
#     'wristright', 'hipleft', 'hipright', 'kneeleft', 'kneeright',
#     'ankleleft','ankleright'
# ]

# key_node_name = [
#     'HeadTop', 'Nose', 'LeftEye', 'RightEye', 'LeftEar', 'RightEar',
#     'LeftShoulder', 'RightShoulder','LeftElbow', 'RightEllbow', 'LeftWrist',
#     'RightWrist', 'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee',
#     'LeftAnkle', 'RightAnkle'
# ]


def cv_draw_node(img, key_node, node1_name, node2_name, color, thickness):
    if (node1_name in key_node) and (node2_name in key_node):
        pt1 = (int(key_node[node1_name][0]), int(key_node[node1_name][1]))
        pt2 = (int(key_node[node2_name][0]), int(key_node[node2_name][1]))
        cv2.line(img, pt1, pt2, color, thickness)


def json_anno_test(img_dir, anno_dir, save_dir):
    for img_file in os.listdir(img_dir):
        if img_file[-3:] != "jpg" and img_file[-3:] != "png":
            continue

        img_path = os.path.join(img_dir, img_file)
        anno_path = os.path.join(anno_dir, img_file[:-3] + "json")
        if not os.path.exists(anno_path):
            print("No annotation:", anno_path)
            continue

        save_path = os.path.join(save_dir, img_file)

        # img = imread(img_path)
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img).astype(np.uint8)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        if img is None:
            str_out = "Unsecceed in processing image: " + img_path
            raise ValueError(str_out)

        try:
            anno_data = file_load(anno_path)

            bbox_list = []
            label_list = []
            keynode_list = []

            for obj in anno_data["object"]:
                # process with each object
                name = obj["name"]
                label = cat2label[name]
                label_list.append(label)

                bnd_box = obj["bndbox"]
                bbox = [
                    int(eval(bnd_box["xmin"])),
                    int(eval(bnd_box["ymin"])),
                    int(eval(bnd_box["xmax"])),
                    int(eval(bnd_box["ymax"])),
                ]
                bbox_list.append(bbox)

                if name == "person":
                    key_nodes = {}
                    # the key node may be lost, we just keep the labelled node
                    if "keynode" in obj:
                        for item in obj["keynode"]:
                            key_nodes[item] = eval(obj["keynode"][item])

                        keynode_list.append(key_nodes)

            bbox_np = np.array(bbox_list)
            label_np = np.array(label_list)
            imshow_det_bboxes(
                img, bbox_np, label_np, class_names=class_names, show=False
            )

            for key_node in keynode_list:
                for key, value in key_node.items():
                    cv2.circle(
                        img,
                        (int(value[0]), int(value[1])),
                        3,
                        (0, 0, 255),
                        thickness=-1,
                    )

                cv_draw_node(
                    img, key_node, key_node_name[0], key_node_name[1], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[1], key_node_name[2], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[1], key_node_name[3], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[2], key_node_name[4], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[3], key_node_name[5], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[6], key_node_name[7], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[6], key_node_name[8], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[8], key_node_name[10], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[7], key_node_name[9], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[9], key_node_name[11], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[12], key_node_name[13], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[12], key_node_name[14], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[14], key_node_name[16], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[13], key_node_name[15], (0, 0, 255), 1
                )
                cv_draw_node(
                    img, key_node, key_node_name[15], key_node_name[17], (0, 0, 255), 1
                )

            # imshow(img, "img_det_anno", 0)
            imwrite(img, save_path)
        except Exception:
            print("---------------------------------------------------------")
            print("Exeception information:")
            print(traceback.format_exc())
            print("Related Path:", anno_path)
            print("---------------------------------------------------------")


if __name__ == "__main__":
    json_anno_test()
