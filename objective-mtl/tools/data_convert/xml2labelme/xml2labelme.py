import os
import sys
import glob
import shutil
import argparse

import cv2
import xmltodict
import itertools
import pypinyin
import json
import yaml
import multiprocessing

import pandas as pd
from tqdm import tqdm
from PIL import Image

import configs as configs


def excel_to_list(excel_label_file, label_ref_dict, excel_title_map) -> list:
    try:
        excel_title_list = list(excel_title_map.values())
        label_df = pd.read_excel(
            excel_label_file, usecols=excel_title_list, dtype="str"
        )
        label_df.columns = list(excel_title_map.keys())
        label_df[["label", "x_min", "y_min", "box_width", "box_height"]] = label_df[
            ["label", "x_min", "y_min", "box_width", "box_height"]
        ].apply(pd.to_numeric)
        label_list = label_df.to_dict("records")
        if label_ref_dict:
            for label_item in label_list:
                try:
                    label_item["label"] = label_ref_dict[label_item["label"]]
                except:
                    label_item["label"] = str(label_item["label"])
                    continue
        return label_list
    except Exception as e:
        raise e


def csv_to_list(csv_label_file) -> list:
    label_df = pd.read_csv(csv_label_file, header=None)
    label_list = label_df.to_dict("records")
    return label_list


def key_label_process(weakness_item) -> dict:
    try:
        weakness_item["label"] = pypinyin.slug(
            weakness_item["label"], separator=""
        ).split(".")[0]
    except:
        pass
    if weakness_item.get("plevel"):
        weakness_item["plevel"] = pypinyin.slug(weakness_item["plevel"], separator="")
    if weakness_item.get("describe"):
        weakness_item["describe"] = pypinyin.slug(
            weakness_item["describe"], separator=""
        )
    if weakness_item.get("width"):
        weakness_item["width"] = int(weakness_item["width"])
    if weakness_item.get("group_id") and weakness_item["group_id"] == "null":
        weakness_item["group_id"] = None
    return weakness_item


def read_complete_image(img_path):
    # complete_flag = False
    # with open(img_path, 'rb') as f:
    #     if img_path.lower().endswith('jpg') or img_path.lower().endswith('jpeg'):
    #         check_chars = f.read()[-2:]
    #         complete_flag = (check_chars == b'\xff\xd9')
    #     elif img_path.lower().endswith('png'):
    #         f.seek(-3, 2)
    #         check_chars = f.read()
    #         complete_flag = (check_chars == b'\x60\x82\x00' or check_chars[1:] == b'\x60\x82')
    # if not complete_flag:
    #     try:
    #         img = Image.open(img_path)
    #         img.load()
    #     except Exception as e:
    #         raise Exception('Not complete image: {}'.format(img_path))
    # else:
    return cv2.imread(img_path, 1)


def process_bndbox(weakness_item):
    try:
        warning_list = []
        weakness_reorg_dict = {
            "label": pypinyin.slug(weakness_item.pop("name"), separator=""),
            "shape_type": "rectangle",
        }
        points_dict = weakness_item.pop("bndbox")
        points_dict = {k: float(v) for k, v in points_dict.items()}
        weakness_reorg_dict["points"] = [
            [points_dict["xmin"], points_dict["ymin"]],
            [points_dict["xmax"], points_dict["ymax"]],
        ]
        if weakness_item.get("width"):
            weakness_item.pop("width")
        weakness_item = key_label_process(weakness_item)
        weakness_reorg_dict.update(weakness_item)
        return weakness_reorg_dict, warning_list
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise e


def process_point(weakness_item):
    try:
        warning_list = []
        weakness_reorg_dict = {
            "label": pypinyin.slug(weakness_item.pop("name"), separator=""),
            "shape_type": "circle",
        }
        points_dict = weakness_item.pop("point")
        points_dict = {k: float(v) for k, v in points_dict.items()}
        if not weakness_item.get("width"):
            weakness_item["width"] = 6
            warning_list.append(KeyError("width"))
        weakness_reorg_dict["points"] = [
            [points_dict["x"], points_dict["y"]],
            [points_dict["x"], points_dict["y"] + float(weakness_item["width"]) // 2],
        ]
        if weakness_item.get("width"):
            weakness_item.pop("width")
        weakness_item = key_label_process(weakness_item)
        weakness_reorg_dict.update(weakness_item)
        return weakness_reorg_dict, warning_list
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise e


def process_ellipse(weakness_item):
    try:
        warning_list = []
        weakness_reorg_dict = {
            "label": pypinyin.slug(weakness_item.pop("name"), separator=""),
            "shape_type": "rectangle",
        }
        points_dict = weakness_item.pop("ellipse")
        points_dict = {k: float(v) for k, v in points_dict.items()}
        weakness_reorg_dict["points"] = [
            [points_dict["xmin"], points_dict["ymin"]],
            [points_dict["xmax"], points_dict["ymax"]],
        ]
        if weakness_item.get("width"):
            weakness_item.pop("width")
        weakness_item = key_label_process(weakness_item)
        weakness_reorg_dict.update(weakness_item)
        return weakness_reorg_dict, warning_list
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise e


def process_polygon(weakness_item):
    try:
        warning_list = []
        weakness_reorg_dict = {
            "label": pypinyin.slug(weakness_item.pop("name"), separator=""),
            "shape_type": "polygon",
        }
        if weakness_item["polygon"].get("points"):
            del weakness_item["polygon"]["points"]
        points_dict = weakness_item.pop("polygon")
        points_dict = {k: float(v) for k, v in points_dict.items()}
        weakness_reorg_dict["points"] = [
            list(points_dict.values())[i : i + 2] for i in range(0, len(points_dict), 2)
        ]
        if weakness_item.get("width"):
            weakness_item.pop("width")
        weakness_item = key_label_process(weakness_item)
        weakness_reorg_dict.update(weakness_item)
        return weakness_reorg_dict, warning_list
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise e


def process_line(weakness_item):
    try:
        warning_list = []
        weakness_reorg_dict = {
            "label": pypinyin.slug(weakness_item.pop("name"), separator="")
        }
        points_dict = weakness_item.pop("line")
        points_dict = {k: float(v) for k, v in points_dict.items()}
        # gen a list consisted of pairs of points
        points_pair_list = [
            list(points_dict.values())[i : i + 2] for i in range(0, len(points_dict), 2)
        ]
        # remove the duplicated pairs which is close to each other
        points_pair_list = [k for k, g in itertools.groupby(points_pair_list)]
        if len(points_pair_list) == 1:
            weakness_item["point"] = {
                "x": points_pair_list[0][0],
                "y": points_pair_list[0][1],
            }
            return process_point(weakness_item)
        else:
            weakness_reorg_dict["shape_type"] = (
                "line" if len(points_pair_list) == 2 else "linestrip"
            )
            weakness_reorg_dict["points"] = points_pair_list
            weakness_item = key_label_process(weakness_item)
            weakness_reorg_dict.update(weakness_item)
            return weakness_reorg_dict, warning_list
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise e


label_kind_to_opt = {
    "bndbox": process_bndbox,
    "point": process_point,
    "ellipse": process_ellipse,
    "polygon": process_polygon,
    "line": process_line,
}


def image_filter(file_list) -> list:
    image_list = []
    for file_item in file_list:
        if os.path.splitext(file_item)[-1].lower() in configs.IMG_SUFFIX_LIST:
            image_list.append(file_item)
    return image_list


class GenLabelme:
    def __init__(self, work_dir, output_path, cfg):
        self.work_dir = work_dir
        self.output_path = os.path.join(output_path, "labelme")
        self.dry_run = cfg.dry_run
        self.disable_jpg = cfg.disable_jpg
        self.recursive = cfg.recursive
        self.excel_file = cfg.excel_file if cfg.excel_file else None
        self.csv_file = cfg.csv_file if cfg.csv_file else None
        self.map_config = cfg.map_config if cfg.map_config else None
        self.process_num = cfg.process_num if cfg.process_num else 8
        self.label_stat_dict = multiprocessing.Manager().dict()
        self.stat_lock = multiprocessing.Manager().Lock()
        try:
            if self.dry_run and os.path.exists(self.output_path):
                shutil.rmtree(self.output_path)
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
        except Exception as e:
            print("Failed to prepare the output path. ERROR: {}.".format(str(e)))
            sys.exit()

    def update_stat_dict(self, label_name):
        label_name = pypinyin.slug(label_name, separator="").split(".")[0]
        self.stat_lock.acquire()
        if label_name in self.label_stat_dict:
            self.label_stat_dict[label_name] += 1
        else:
            self.label_stat_dict[label_name] = 1
        self.stat_lock.release()

    @staticmethod
    def get_data_list(work_dir_list) -> dict:
        data_list = {"xml_list": [], "json_list": []}
        exception_list = []
        for work_dir in work_dir_list:
            exception_flag = 0
            if os.path.isdir(os.path.join(work_dir, "outputs")) and glob.glob(
                os.path.join(work_dir, "outputs", "*.xml")
            ):
                data_list["xml_list"].append(work_dir)
                exception_flag += 1
            if glob.glob(os.path.join(work_dir, "*.json")):
                data_list["json_list"].append(work_dir)
                exception_flag += 1
            if exception_flag > 1:
                exception_list.append(work_dir)
        if exception_list:
            print(
                "WARNING: These folder contains more than 1 kind of format data, {}".format(
                    str(exception_list)
                )
            )
        return data_list

    def xml_to_labelme(self, source_data_path: str, data_item: str):
        data_abs_path = os.path.join(source_data_path, data_item)
        data_suffix = os.path.splitext(data_abs_path)[-1]
        data_id = data_item.rstrip(data_suffix)
        xml_file_path = os.path.join(source_data_path, "outputs", data_id + ".xml")
        if (
            os.path.isfile(data_abs_path)
            and data_suffix.lower() in configs.IMG_SUFFIX_LIST
        ):
            try:
                labelme_json_out_file = os.path.join(
                    self.output_path, data_id + ".json"
                )
                with open(xml_file_path, encoding="utf-8") as xml_file:
                    xml_label = xml_file.read()
                xml_label_dict = xmltodict.parse(xml_label)["doc"]
                img_size = {k: int(v) for k, v in xml_label_dict["size"].items()}
                img_name = (
                    data_id + ".jpg"
                    if not self.disable_jpg
                    else os.path.basename(data_abs_path)
                )
                labelme_json_instance = configs.gen_labelme_json_model(
                    img_size, img_name
                )
                weakness_list = xml_label_dict["outputs"]["object"]["item"]
                weakness_list = (
                    weakness_list
                    if isinstance(weakness_list, list)
                    else [weakness_list]
                )
                for weakness_item in weakness_list:
                    label_kind = label_kind_to_opt.keys() & weakness_item.keys()
                    assert (
                        len(label_kind) == 1
                    ), "ERROR: Fail to read the {} in {}.".format(
                        str(weakness_item), xml_file_path
                    )
                    label_kind = label_kind.pop()
                    self.update_stat_dict(weakness_item.get("name"))
                    shapes_result_item, warning_list = label_kind_to_opt[label_kind](
                        weakness_item
                    )
                    if warning_list:
                        print(
                            "WARNING: {} occured while processing {}.".format(
                                xml_file_path, str(warning_list)
                            )
                        )
                    labelme_json_instance["shapes"].append(shapes_result_item)
                with open(labelme_json_out_file, "w", encoding="utf-8") as out_file:
                    json.dump(
                        labelme_json_instance, out_file, ensure_ascii=False, indent=2
                    )
                if not self.disable_jpg:
                    img_file = read_complete_image(data_abs_path)
                    cv2.imwrite(
                        os.path.join(self.output_path, data_id + ".jpg"),
                        img_file,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                    )
                else:
                    shutil.copy(data_abs_path, self.output_path)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(
                    "Fail to convert the data {}, ERROR: {}.".format(
                        xml_file_path, str(e)
                    )
                )
                if os.path.exists(os.path.join(self.output_path, data_id + ".jpg")):
                    os.remove(os.path.join(self.output_path, data_id + ".jpg"))
                if os.path.exists(os.path.join(self.output_path, data_id + ".json")):
                    os.remove(os.path.join(self.output_path, data_id + ".json"))

    @staticmethod
    def collect_json_format_data(source_data_path: str):
        data_list = os.listdir(source_data_path)
        img_dict = {}
        json_dict = {}
        for data_item in data_list:
            data_abs_path = os.path.join(source_data_path, data_item)
            data_suffix = os.path.splitext(data_abs_path)[-1]
            data_id = data_item.rstrip(data_suffix)
            if (
                os.path.isfile(data_abs_path)
                and data_suffix.lower() in configs.IMG_SUFFIX_LIST
            ):
                img_dict[data_id] = data_abs_path
            if os.path.isfile(data_abs_path) and data_suffix.lower() == ".json":
                json_dict[data_id] = data_abs_path
        return img_dict, json_dict

    def json_to_labelme(self, img_dict: dict, data_id: str, json_item):
        if img_dict.get(data_id):
            try:
                with open(json_item) as input_file:
                    label_json = json.load(input_file)
                assert (
                    len(
                        label_json.keys()
                        & configs.gen_labelme_json_model(None, None).keys()
                    )
                    >= 4
                ), "The format of json file {} is not correct.".format(json_item)
                for label_item in label_json.get("shapes", []):
                    label_item = key_label_process(label_item)
                    self.update_stat_dict(label_item["label"])
                label_json["imageData"] = None
                label_json["imagePath"] = (
                    data_id + ".jpg"
                    if not self.disable_jpg
                    else os.path.basename(img_dict[data_id])
                )
                with open(
                    os.path.join(self.output_path, data_id + ".json"),
                    "w",
                    encoding="utf-8",
                ) as out_file:
                    json.dump(label_json, out_file, ensure_ascii=False, indent=2)
                if not self.disable_jpg:
                    img_file = read_complete_image(img_dict[data_id])
                    cv2.imwrite(
                        os.path.join(self.output_path, data_id + ".jpg"),
                        img_file,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                    )
                else:
                    shutil.copy(img_dict[data_id], self.output_path)
            except Exception as e:
                print(
                    "Fail to convert the data {}, ERROR: {}.".format(json_item, str(e))
                )
                if os.path.exists(os.path.join(self.output_path, data_id + ".jpg")):
                    os.remove(os.path.join(self.output_path, data_id + ".jpg"))
                if os.path.exists(os.path.join(self.output_path, data_id + ".json")):
                    os.remove(os.path.join(self.output_path, data_id + ".json"))

    def collect_excel_detail(self, map_config, excel_file) -> dict:
        try:
            with open(map_config) as input_file:
                excel_map_dict = yaml.load(input_file, Loader=yaml.FullLoader)
            label_ref_dict = excel_map_dict.pop("label_dict")
            excel_title_map = excel_map_dict
            label_list = excel_to_list(excel_file, label_ref_dict, excel_title_map)
            print(label_list)
            labelme_shape_dict = {}

            for label_item in label_list:
                if (
                    len(label_item["task_id"]) <= 4
                    and len(label_item["product_sn"]) <= 4
                    and len(label_item["picture_id"]) <= 2
                ):
                    data_id = "{}-{}-{}".format(
                        label_item["task_id"].rjust(4, "0"),
                        label_item["product_sn"].rjust(4, "0"),
                        label_item["picture_id"].rjust(2, "0"),
                    )
                    print(data_id)
                    self.update_stat_dict(label_item.get("label"))
                    label_shape_item = {
                        "label": pypinyin.slug(label_item.pop("label"), separator=""),
                        "shape_type": "rectangle",
                        "points": [
                            [label_item["x_min"], label_item["y_min"]],
                            [
                                label_item["x_min"] + label_item["box_width"],
                                label_item["y_min"] + label_item["box_height"],
                            ],
                        ],
                    }
                    if labelme_shape_dict.get(data_id):
                        labelme_shape_dict[data_id].append(label_shape_item)
                    else:
                        labelme_shape_dict[data_id] = [label_shape_item]
                else:
                    raise Exception("Failed to convert {}.".format(str(label_item)))
            return labelme_shape_dict
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(
                "Failed to collect info from {}. WARNING: {}".format(excel_file, str(e))
            )
            return {}

    @staticmethod
    def convert_tianchi_geometry(geometry_list):
        point_1 = [geometry_list[0], geometry_list[1]]
        point_2 = [geometry_list[2], geometry_list[3]]
        return (
            [point_1, point_2]
            if geometry_list[0] < geometry_list[2]
            else [point_2, point_1]
        )

    def collect_csv_detail(self, csv_file) -> dict:
        try:
            label_list = csv_to_list(csv_file)
            labelme_shape_dict = {}

            for label_item in label_list:
                if label_item[4] and label_item[5]:
                    img_name = os.path.split(label_item[4])[-1]
                    data_suffix = os.path.splitext(img_name)[-1]
                    data_id = img_name.rstrip(data_suffix)
                    label_json = json.loads(label_item[5])
                    for shape_item in label_json.get("items"):
                        self.update_stat_dict(shape_item.get("labels").get("标签"))
                        label_shape_item = {
                            "label": shape_item.get("labels").get("标签"),
                            "shape_type": "rectangle",
                            "points": GenLabelme.convert_tianchi_geometry(
                                shape_item.get("meta").get("geometry")
                            ),
                        }
                        if labelme_shape_dict.get(data_id):
                            labelme_shape_dict[data_id].append(label_shape_item)
                        else:
                            labelme_shape_dict[data_id] = [label_shape_item]
                else:
                    raise Exception("Failed to convert {}.".format(str(label_item)))
            return labelme_shape_dict
        except Exception as e:
            print(
                "Failed to collect info from {}. WARNING: {}".format(csv_file, str(e))
            )
            return {}

    # source_data_path consisted of images
    def assemble_labelme(self, source_data_path: str, data_id: str, shape_item):
        try:
            img_file_path = image_filter(
                glob.glob(os.path.join(source_data_path, data_id + "*"))
            )
            if len(img_file_path) > 1:
                raise Exception(
                    "Multiple image files detected for ID: {}.".format(data_id)
                )
            elif len(img_file_path) == 0:
                raise Exception("No image files detected for ID: {}.".format(data_id))
            else:
                img_file_path = img_file_path.pop()
                img_file = read_complete_image(img_file_path)
                img_size = {
                    "width": img_file.shape[1],
                    "height": img_file.shape[0],
                    "depth": img_file.shape[2],
                }
                img_name = (
                    data_id + ".jpg"
                    if not self.disable_jpg
                    else os.path.basename(img_file_path)
                )
                label_json = configs.gen_labelme_json_model(img_size, img_name)
                label_json["shapes"].extend(shape_item)
                with open(
                    os.path.join(self.output_path, data_id + ".json"),
                    "w",
                    encoding="utf-8",
                ) as out_file:
                    json.dump(label_json, out_file, ensure_ascii=False, indent=2)
                if not self.disable_jpg:
                    cv2.imwrite(
                        os.path.join(self.output_path, data_id + ".jpg"),
                        img_file,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
                    )
                else:
                    shutil.copy(img_file_path, self.output_path)
        except Exception as e:
            print(
                "Fail to convert the data {}, WARNING: {}.".format(
                    self.excel_file, str(e)
                )
            )

    def trigger_generate(self):
        if self.excel_file:
            data_list = {"excel_list": [self.work_dir]}
        elif self.csv_file:
            data_list = {"csv_list": [self.work_dir]}
        else:
            work_dir_list = (
                glob.glob(os.path.join(self.work_dir, "*"))
                if self.recursive
                else [self.work_dir]
            )
            data_list = self.get_data_list(work_dir_list)
        # process the labels with xml format
        if data_list.get("xml_list"):
            xml_pool = multiprocessing.Pool(processes=self.process_num)
            for xml_source_path in tqdm(data_list["xml_list"], desc="xml_list"):
                img_data_list = os.listdir(xml_source_path)
                process_bar = tqdm(total=len(img_data_list))
                for img_data_item in img_data_list:
                    xml_pool.apply_async(
                        self.xml_to_labelme,
                        args=(xml_source_path, img_data_item),
                        callback=lambda _: process_bar.update(),
                    )
            xml_pool.close()
            xml_pool.join()
        # process the labels with excel format
        if data_list.get("excel_list"):
            excel_pool = multiprocessing.Pool(processes=self.process_num)
            for excel_path in tqdm(data_list["excel_list"]):
                labelme_shape_dict = self.collect_excel_detail(
                    self.map_config, self.excel_file
                )
                print(labelme_shape_dict)
                if labelme_shape_dict:
                    process_bar = tqdm(total=len(labelme_shape_dict))
                    for data_id, shape_item in labelme_shape_dict.items():
                        excel_pool.apply_async(
                            self.assemble_labelme,
                            args=(excel_path, data_id, shape_item),
                            callback=lambda _: process_bar.update(),
                        )
            excel_pool.close()
            excel_pool.join()
        # process the labels with json format
        if data_list.get("json_list"):
            json_pool = multiprocessing.Pool(processes=self.process_num)
            for json_path in tqdm(data_list["json_list"], desc="json_list"):
                img_dict, json_dict = self.collect_json_format_data(json_path)

                if not json_dict or not img_dict:
                    raise Exception("{} contains no img/json".format(json_path))
                else:
                    process_bar = tqdm(total=len(json_dict))
                    for data_id, json_item in json_dict.items():
                        json_pool.apply_async(
                            self.json_to_labelme,
                            args=(img_dict, data_id, json_item),
                            callback=lambda _: process_bar.update(),
                        )
            json_pool.close()
            json_pool.join()
        # process the labels with the TianChi csv format
        if data_list.get("csv_list"):
            csv_pool = multiprocessing.Pool(processes=self.process_num)
            for csv_path in tqdm(data_list["csv_list"]):
                labelme_shape_dict = self.collect_csv_detail(self.csv_file)
                if labelme_shape_dict:
                    process_bar = tqdm(total=len(labelme_shape_dict))
                    for data_id, shape_item in labelme_shape_dict.items():
                        csv_pool.apply_async(
                            self.assemble_labelme,
                            args=(csv_path, data_id, shape_item),
                            callback=lambda _: process_bar.update(),
                        )
            csv_pool.close()
            csv_pool.join()
        print("========================================")
        print(self.label_stat_dict)
        print("========================================")
        print("SUCCESS")


def get_parser():
    parser = argparse.ArgumentParser(
        description="The tool used to convert the label of standard labelme format to yolo_text."
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        required=True,
        type=str,
        default=None,
        help="The path of the work_dir.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="The path to save the output. (default: work_dir/../labelme, required if convert excel format label)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan the work_dir recursively. (default: False)",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="Whether run without cleaning the output path. (default: False)",
    )
    parser.add_argument(
        "--disable_jpg",
        action="store_true",
        help="Whether disable the function of converting origin image to the JPEG format. (default: False)",
    )
    parser.add_argument(
        "-x",
        "--excel_file",
        type=str,
        default=None,
        help="The label file stored with excel format. (Required if convert excel format label)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="The label file stored with csv format. (For the Tianchi challenge)",
    )
    parser.add_argument(
        "-c",
        "--map_config",
        type=str,
        default=None,
        help="The config file for decode the excel file, the format could refer to map_config.yml. (Required if convert excel format label)",
    )
    parser.add_argument(
        "-p",
        "--process_num",
        type=int,
        default=8,
        help="The num of workers for multiprocess. (default: 8)",
    )
    opt = parser.parse_args()

    if not os.path.isdir(opt.work_dir):
        print(
            "ERROR: The work_dir given ({}) is not a valid path.".format(opt.work_dir)
        )
        sys.exit()
    if opt.excel_file:
        if not opt.output_path:
            print(
                "ERROR: The output_path is required while converting.".format(
                    opt.output_path
                )
            )
            sys.exit()
        if not opt.map_config:
            print(
                "ERROR: The map_config is required while converting.".format(
                    opt.excel_file
                )
            )
            sys.exit()
    opt.output_path = (
        opt.output_path
        if opt.output_path
        else os.path.abspath(os.path.join(opt.work_dir, ".."))
    )
    return opt


if __name__ == "__main__":
    args = get_parser()
    label_process = GenLabelme(args.work_dir, args.output_path, args)
    label_process.trigger_generate()
