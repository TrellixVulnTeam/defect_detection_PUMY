import os
import shutil
import numpy as np
from PIL import Image
import traceback

from mtl.utils.io_util import file_load


def anno_check_and_move(anno_file, img_dir, move_dir, new_anno_file, split_str=" "):
    """Check the annotated images and move unannotated ones to move_dir"""
    file_infos = []
    with open(anno_file, "r") as rf:
        lines = rf.readlines()
        for line in lines:
            tmp_str = line.strip()
            str_list = tmp_str.split(split_str)
            if len(str_list) >= 2:
                img_file = os.path.basename(str_list[0])
                file_infos.append(img_file)

    # check the image
    valid_img_files = []
    for img_file in os.listdir(img_dir):
        if img_file not in file_infos:
            shutil.move(os.path.join(img_dir, img_file), move_dir)
        valid_img_files.append(img_file)

    # check the annotations
    with open(new_anno_file, "w") as rw:
        with open(anno_file, "r") as rf:
            lines = rf.readlines()
        for line in lines:
            tmp_str = line.strip()
            str_list = tmp_str.split(split_str)
            if len(str_list) >= 2:
                img_file = os.path.basename(str_list[0])
                if img_file in valid_img_files:
                    try:
                        pil_img = Image.open(os.path.join(img_dir, img_file))
                        rw.write(
                            "%s %s %d %d\n"
                            % (img_file, str_list[1], pil_img.size[0], pil_img.size[1])
                        )
                    except Exception:
                        print(
                            "---------------------------------------------------------"
                        )
                        print("Exeception information:")
                        print(traceback.format_exc())
                        print("Related Path:", img_file)
                        print(
                            "---------------------------------------------------------"
                        )


def remove_same_labels(anno_file, out_file):
    with open(anno_file, "r") as rf:
        lines = rf.readlines()
    data_infos = []
    with open(out_file, "w") as rw:
        for line in lines:
            tmp_str = line.strip()
            if tmp_str in data_infos:
                continue
            data_infos.append(tmp_str)
            rw.write(line)


def train_val_split(ann_file, train_file, val_file, percent=10):
    """Split the annotation file into a train file and a valid file"""
    twf = open(train_file, "w")
    vwf = open(val_file, "w")
    with open(ann_file, "r") as rf:
        lines = rf.readlines()
        for line in lines:
            tmp_str = line.strip()
            chance = np.random.randint(100)
            if chance < 100 - percent:
                twf.write(tmp_str + "\n")
            else:
                vwf.write(tmp_str + "\n")


def json_file_convert(json_dir, reanno_dir):
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            try:
                anno_data = file_load(os.path.join(json_dir, json_file))
                img_name = anno_data["asset"]["name"]

                with open(os.path.join(reanno_dir, img_name[:-3] + "txt"), "w") as rw:
                    for region in anno_data["regions"]:
                        label_name = region["tags"][0]
                        xmin = int(region["boundingBox"]["left"] + 0.5)
                        ymin = int(region["boundingBox"]["top"] + 0.5)
                        xmax = int(
                            region["boundingBox"]["left"]
                            + region["boundingBox"]["width"]
                            + 0.5
                        )
                        ymax = int(
                            region["boundingBox"]["top"]
                            + region["boundingBox"]["height"]
                            + 0.5
                        )

                        rw.write(
                            "%s %s %s %s %s\n" % (label_name, xmin, ymin, xmax, ymax)
                        )

            except Exception:
                print("---------------------------------------------------------")
                print("Exeception information:")
                print(traceback.format_exc())
                print("Related Path:", json_file)
                print("---------------------------------------------------------")


def file_check_move(anno_dir, img_dir, unannotated_dir, related_image_dir):
    for anno_file in os.listdir(anno_dir):
        if anno_file.endswith(".txt"):
            try:
                image_name = anno_file[:-3] + "jpg"
                img_path = os.path.join(img_dir, image_name)
                if os.path.exists(img_path):
                    shutil.move(img_path, related_image_dir)
                else:
                    shutil.move(os.path.join(anno_dir, anno_file), unannotated_dir)

            except Exception:
                print("---------------------------------------------------------")
                print("Exeception information:")
                print(traceback.format_exc())
                print("Related Path:", anno_file)
                print("---------------------------------------------------------")


def file_move(src_dir, img_dir, anno_dir, dst_img_dir, dst_anno_dir):

    for src_file in os.listdir(src_dir):
        print(src_file)
        if src_file.endswith(".jpg"):
            try:
                img_path = os.path.join(img_dir, src_file)
                anno_path = os.path.join(anno_dir, src_file[:-3] + "xml")

                if os.path.exists(img_path):
                    shutil.move(img_path, dst_img_dir)

                if os.path.exists(anno_path):
                    shutil.move(anno_path, dst_anno_dir)

            except Exception:
                print("---------------------------------------------------------")
                print("Exeception information:")
                print(traceback.format_exc())
                print("Related Path:", src_file)
                print("---------------------------------------------------------")


def change_name2label(anno_dir, reanno_dir, cat2label):
    for label_file in os.listdir(anno_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(anno_dir, label_file), "r") as fr:
                lines = fr.readlines()

            with open(os.path.join(reanno_dir, label_file), "w") as fw:
                for line in lines:
                    if line.strip() != "":
                        tmp_list = line.split()

                        label = cat2label[tmp_list[0]]

                        fw.write(
                            "%d %s %s %s %s\n"
                            % (
                                label,
                                tmp_list[1],
                                tmp_list[2],
                                tmp_list[3],
                                tmp_list[4],
                            )
                        )


if __name__ == "__main__":
    print("Remove the same labels from the annotation file")
    anno_file = "xxx"
    out_file = "xxx.txt"
    remove_same_labels(anno_file, out_file)

    print(
        "Check the corresponding image files, if not exist, move to the unannotated folder"
    )
    anno_file = "xxx.txt"
    img_dir = "xxx"
    move_dir = "xxx"
    new_anno_file = "xxx.txt"
    anno_check_and_move(anno_file, img_dir, move_dir, new_anno_file, " ")

    print(
        "Split the train and val lists with the certain percetage, default 90 for train, 10 for validate"
    )
    anno_file = "xxx.txt"
    train_file = "xxx.txt"
    val_file = "xxx.txt"
    train_val_split(anno_file, train_file, val_file)

    print("Generating annotation file")
    json_dir = "xxx{json_dir}"
    anno_dir = "xxx"
    json_file_convert(json_dir, anno_dir)

    print("Check the corresponding annotation files")
    anno_dir = "xxx"
    img_dir = "xxx"
    unannotated_dir = "xxx"
    related_image_dir = "xxx"
    file_check_move(anno_dir, img_dir, unannotated_dir, related_image_dir)

    anno_dir = "xxx"
    reanno_dir = "xxx"
    cat2label = {cat: i for i, cat in enumerate(("cat", "dog"))}
    change_name2label(anno_dir, reanno_dir, cat2label)

    src_dir = "xxx"
    img_dir = "xxx"
    anno_dir = "xxx"
    dst_img_dir = "xxx"
    dst_anno_dir = "xxx"
    file_move(src_dir, img_dir, anno_dir, dst_img_dir, dst_anno_dir)
