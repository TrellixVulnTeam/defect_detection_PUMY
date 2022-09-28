import argparse
import json
import os
import shutil

import numpy
import multiprocessing

import pypinyin
from tqdm import tqdm
import pycocotools.mask
import cv2
import labelme
import glob
import sys
import yaml

from wycv.label_process.label_converter.train_val_split import random_split, filter_split
from wycv.label_process.label_converter.COCOFactory import COCOFactory
from wycv.label_process.label_converter.YOLOFactory import YOLOFactory


class LabelConvert:
    def __init__(self, cfg):
        self.work_dir = cfg.get('work_dir')
        self.label_dict = multiprocessing.Manager().dict(cfg.get('label_dict', {}))  # key: label_name, value: label_id
        self.label_sort_flag = (len(self.label_dict) == 0)
        self.label_dict_lock = multiprocessing.Manager().Lock()
        self.target_format = cfg['convert_params'].pop('target_format')
        self.output_path = os.path.join(cfg.get('output_path'), self.target_format)
        self.convert_params = cfg['convert_params']
        self.remain_bg_flag = self.convert_params.get('remain_bg')
        self.process_num = cfg['process_num']
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        self.img_id_dict = {}  # key: img_id, value: img_item of coco_format FOR COCO
        self.img_mask_dict = multiprocessing.Manager().dict()

        self.divide_method = cfg['split_params']['method'] if cfg.get('split_params') else None
        self.split_params = cfg['split_params'] if self.divide_method else None
        self.val_ratio = cfg['split_params']['val_ratio'] if cfg.get('split_params') else 0
        self.test_ratio = cfg['split_params']['test_ratio'] if cfg.get('split_params') else 0

    def update_label_dict(self, label_item):
        label_name = pypinyin.slug(label_item, separator='')
        self.label_dict_lock.acquire()
        if label_name not in self.label_dict:
            self.label_dict[label_name] = len(self.label_dict)
        self.label_dict_lock.release()

    def trigger_convert(self):
        if self.target_format == 'coco':
            return self.trigger_convert_for_coco()
        elif self.target_format == 'yolo':
            return self.trigger_convert_for_yolo()
        else:
            print('ERROR: Unknown target format: {}.'.format(self.target_format))
            sys.exit()

    def trigger_convert_for_coco(self):
        image_anno_dict = {}
        shape_list = []
        json_list = glob.glob(os.path.join(self.work_dir, '*.json'))
        img_idx = 0
        for json_item in tqdm(json_list, desc='Reading source data ...', total=len(json_list)):
            with open(json_item, 'r', encoding='utf-8') as infile:
                json_data = json.load(infile)
                if json_data['shapes']:
                    img_item = LabelConvert.image_build(json_data, img_idx, json_item)
                    self.img_id_dict[img_idx] = img_item
                    image_anno_dict = self.update_image_anno_dict(image_anno_dict, json_data['shapes'], img_idx)
                    shape_list.extend(json_data['shapes'])
                    img_idx += 1

        label_sorted_list = sorted(list(self.label_dict.keys())) if self.label_sort_flag else list(
            self.label_dict.keys())

        if self.remain_bg_flag:
            label_sorted_list.insert(0, 'background')
            self.label_dict = {label_name: idx for idx, label_name in enumerate(label_sorted_list)}
        else:
            self.label_dict = {label_name: idx + 1 for idx, label_name in enumerate(label_sorted_list)}
        coco_instance = COCOFactory(list(self.img_id_dict.values()), self.work_dir, self.output_path,
                                    self.convert_params.get('gen_mask', True),
                                    self.convert_params.get('color_mask', True), self.label_sort_flag, self.process_num)
        gen_anno_pool = multiprocessing.Pool(processes=self.process_num)
        shape_pbar = tqdm(total=len(shape_list), desc='Generating the annotation list ...')
        for shape_idx, data_shape in enumerate(shape_list):
            img_idx = image_anno_dict[shape_idx]
            gen_anno_pool.apply_async(self.anno_item_update, args=(coco_instance, img_idx, shape_idx, data_shape,),
                                      callback=lambda _: shape_pbar.update())
        gen_anno_pool.close()
        gen_anno_pool.join()

        coco_instance.img_mask_dict = self.img_mask_dict
        coco_instance.init_category_list(self.label_dict)
        return coco_instance

    def trigger_convert_for_yolo(self):
        label_txt_dict = multiprocessing.Manager().dict()
        json_list = glob.glob(os.path.join(self.work_dir, '*.json'))
        yolo_text_build_pool = multiprocessing.Pool(processes=self.process_num)
        yolo_pbar = tqdm(total=len(json_list))
        for json_idx, json_data in tqdm(enumerate(json_list), desc='Reading source data ...'):
            yolo_text_build_pool.apply_async(self.yolo_text_build, args=(json_data, label_txt_dict,),
                                             callback=lambda _: yolo_pbar.update())
        yolo_text_build_pool.close()
        yolo_text_build_pool.join()
        yolo_instance = YOLOFactory(label_txt_dict, self.work_dir, self.output_path, self.process_num)
        return yolo_instance

    def yolo_text_build(self, json_data, label_txt_dict):
        try:
            with open(json_data, 'r', encoding='utf-8') as f:
                json_instance = json.load(f)
                img_name = os.path.basename(json_instance['imagePath'])
                width, height = json_instance['imageWidth'], json_instance['imageHeight']
                shapes = json_instance['shapes']
                yolo_text_list = []
                for shape in shapes:
                    try:
                        label_id = self.label_dict[shape['label']]
                    except:
                        self.update_label_dict(shape['label'])
                        label_id = self.label_dict[shape['label']]
                    points = numpy.array(shape['points'])
                    if shape['shape_type'] == 'circle':
                        center_point_x, center_point_y = points[0][0] / width, points[0][1] / height
                        box_width, box_height = (numpy.linalg.norm(points[0] - points[1])) * 2 / width, (
                            numpy.linalg.norm(points[0] - points[1])) * 2 / height
                    else:
                        points_x, points_y = points.T
                        min_x, min_y = min(points_x), min(points_y)
                        max_x, max_y = max(points_x), max(points_y)
                        center_point_x, center_point_y = (min_x + max_x) / (2 * width), (min_y + max_y) / (2 * height)
                        box_width, box_height = (max_x - min_x) / width, (max_y - min_y) / height

                    # box_info = "%d %.03f %.03f %.03f %.03f" % (box[1], x_center, y_center, w, h)
                    yolo_text_list.append([label_id, center_point_x, center_point_y, box_width, box_height])
                label_txt_dict[img_name] = yolo_text_list
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('ERROR: {}'.format(str(e)))

    @staticmethod
    def image_build(data, idx, json_name):
        image = {'height': data["imageHeight"],
                 'width': data["imageWidth"],
                 'id': idx,
                 'file_name': os.path.basename(json_name).replace('.json', '.jpg')}
        return image

    def update_image_anno_dict(self, image_anno_dict, shape_list, img_idx):
        start_idx = len(image_anno_dict)
        for shape_idx, shape_item in enumerate(shape_list):
            image_anno_dict[start_idx + shape_idx] = img_idx
            if shape_item['label'] not in self.label_dict:
                self.label_dict[shape_item['label']] = len(self.label_dict)
        return image_anno_dict

    def anno_item_update(self, coco_instance, img_idx, shape_idx, shape_item):
        try:
            img_shape = (self.img_id_dict[img_idx]['height'], self.img_id_dict[img_idx]['width'])
            cate_id = self.label_dict[shape_item['label'].split('_')[0]]
            mask = labelme.utils.shape.shape_to_mask(img_shape, shape_item['points'], shape_item['shape_type'],
                                                     line_width=int(shape_item['width'] if shape_item.get(
                                                         'width') else self.convert_params.get('line_with', 5)))
            try:
                self.img_mask_dict[img_idx].append({'mask': mask, 'category_id': cate_id})
            except:
                self.img_mask_dict[img_idx] = [{'mask': mask, 'category_id': cate_id}]
            annotation = {'bbox': list(map(float, LabelConvert.mask2box(mask)))}
            mask = numpy.asfortranarray(mask).astype('uint8')
            if self.convert_params.get('isRLE'):
                segm = pycocotools.mask.encode(mask)  # 编码为rle格式
                annotation['area'] = float(pycocotools.mask.area(segm))  # 计算mask编码的面积，必须放置在mask转字符串前面，否则计算为0
                segm['counts'] = bytes.decode(segm['counts'])  # 将字节编码转为字符串编码
                annotation['segmentation'] = segm
                annotation['iscrowd'] = 1
            else:
                segm1 = pycocotools.mask.encode(mask)  # 非rle格式
                annotation['area'] = float(pycocotools.mask.area(segm1))  # 计算mask编码的面积，必须放置在mask转字符串前面，否则计算为0
                contours = LabelConvert.get_contours_binary(mask)
                annotation['segmentation'] = [numpy.squeeze(contours[0]).flatten().tolist()] if len(contours) != 0 else \
                shape_item['points']
                annotation['iscrowd'] = 0
            if 'plevel' in shape_item:
                annotation['plevel'] = shape_item['plevel']
            if 'describe' in shape_item:
                annotation['describe'] = shape_item['describe']
            annotation['image_id'] = img_idx
            annotation['category_id'] = cate_id
            annotation['id'] = shape_idx
            coco_instance.add_anno(annotation)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Failed to process {}. ERROR:  {}'.format(coco_instance.image_list[img_idx]['file_name'], str(e)))

    def train_val_divide(self, data_instance):
        total_img_list = data_instance.get_img_id_list()
        if self.divide_method == 'random_split':
            train_list, val_list, test_list = random_split(total_img_list, self.val_ratio, self.test_ratio,
                                                           self.split_params.get('random_seed'))
        elif self.divide_method == 'filter_split':
            train_list, val_list, test_list = filter_split(total_img_list, self.val_ratio, self.test_ratio,
                                                           data_instance,
                                                           self.split_params.get('filter_label',
                                                                                 list(self.label_dict.values())),
                                                           self.split_params.get('level', 3))
        elif not self.divide_method:
            print('WARNING: No split method assigned, all data would be put to training set.')
            train_list, val_list, test_list = total_img_list, [], []
        else:
            raise Exception('ERROR: {} could not be recognize as a valid split method.'.format(self.divide_method))
        data_instance.set_train_val(train_list, val_list, test_list)

    @staticmethod
    def mask2box(mask):
        # 从mask反算出其边框
        # mask：[h,w]  0、1组成的图片
        # 1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        index = numpy.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = numpy.min(rows)  # y
        left_top_c = numpy.min(clos)  # x
        # 解析右下角行列号
        right_bottom_r = numpy.max(rows)
        right_bottom_c = numpy.max(clos)
        return [left_top_c, left_top_r, right_bottom_c - left_top_c, right_bottom_r - left_top_r]  # [x1, y1, w, h]

    @staticmethod
    def get_contours_binary(img):
        ret, binary = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [items for items in contours if len(items) > 2]
        if not polygons:
            raise Exception('Invalid points of polygons: {}'.format(str(contours)))
        return polygons


def get_parser():
    parser = argparse.ArgumentParser(
        description='The tool used to convert the label of standard labelme format to coco or yolo_text format.')
    parser.add_argument('-c', '--config', required=True, type=str, default=None, help='The path of the config file.')
    parser.add_argument('-p', '--process_num', type=int, default=8,
                        help='The num of workers for multiprocess. (default: 8)')
    opt = parser.parse_args()
    try:
        with open(opt.config, 'rb') as input_file:
            config_dict = yaml.load(input_file, Loader=yaml.FullLoader)
            config_dict['process_num'] = opt.process_num
        return config_dict
    except Exception as e:
        print('Failed to open the config file {}. ERROR: {}.'.format(opt.config, str(e)))
        sys.exit()


def params_check(config_dict, config_dict_model, params_error_list):
    for k in set(config_dict_model):
        if k not in config_dict:
            params_error_list.append(k)
            return False
        elif not isinstance(config_dict_model[k], dict):
            continue
        elif params_check(config_dict[k], config_dict_model[k], params_error_list):
            continue
    return True


if __name__ == '__main__':
    args = get_parser()

    cv2.setNumThreads(0)

    label_converter = LabelConvert(args)
    convert_instance = label_converter.trigger_convert()
    label_converter.train_val_divide(convert_instance)
    convert_instance.save_result()
