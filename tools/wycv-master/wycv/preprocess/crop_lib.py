import copy
import math
import os
import random
import itertools
from sklearn.metrics.pairwise import pairwise_distances
from pathlib import Path

import cv2
from labelme.utils.shape import shape_to_mask as lm_shape_to_mask
from wycv.preprocess.configs import ExceedLimitError
import numpy as np


class ContourExtractor:
    def __init__(self, **awk):
        if len(awk) == 2 and set(['shape_item', 'img_shape']).issubset(awk.keys()):
            mask = lm_shape_to_mask(awk['img_shape'], awk['shape_item'].pop('points'),
                                    awk['shape_item'].pop('shape_type'),
                                    line_width=int(awk['shape_item'].get('width', 5)))  # draw mask
            mask = np.asfortranarray(mask).astype('uint8')
            _, self.mask_binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        elif len(awk) == 1 and 'mask' in awk.keys():
            _, self.mask_binary = cv2.threshold(awk['mask'], 0.5, 255, cv2.THRESH_BINARY)
        else:
            raise Exception('Unknown parameters list for ContourExtractor: {}'.format(str(awk.keys())))

    def get_contour(self, crop_loc=None):
        local_mask = self.mask_binary if crop_loc is None else \
            self.mask_binary[crop_loc['start_y']: crop_loc['end_y'], crop_loc['start_x']: crop_loc['end_x']]
        if np.all(local_mask == 0):
            return []
        else:
            local_contours, _ = cv2.findContours(local_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
            return [np.squeeze(items).flatten().tolist() for items in local_contours if len(items) > 2]


def compute_center(point_list):
    center_dict = {}
    for pindex, pitem in enumerate(point_list):
        p_np = np.array(pitem).T
        center_dict[pindex] = [(np.max(p_np[0]) - np.min(p_np[0])) / 2 + np.min(p_np[0]),
                               (np.max(p_np[1]) - np.min(p_np[1])) / 2 + np.min(p_np[1])]
    return center_dict


def del_point(index, array_dict, matrix):
    array_dict.pop(list(array_dict.keys())[index])
    matrix = np.delete(matrix, [index], axis=0)
    matrix = np.delete(matrix, [index], axis=1)
    return array_dict, matrix


def judge_loc(crop_start, crop_length, limitation):
    try:
        if crop_start < 0 and crop_start + crop_length > limitation:
            raise ExceedLimitError('The crop params exceed the limitation.')
        new_start = np.max([0, crop_start])  # min limit
        new_start = np.min([new_start + crop_length, limitation]) - crop_length  # max limit
        return new_start
    except ExceedLimitError:
        # print('Warning: The crop params exceed the limitation.')
        return crop_start


class ClusterCrop:
    """support crop_params:
    bias: bool,
    resample: int > 0,
    mega_instance_policy: choice['center', 'resize']
    """

    def __init__(self, img_item, json_item, output_size, crop_params):
        self.img_item = img_item
        self.json_item = json_item
        self.output_size = output_size
        self.cut_w = self.output_size['width']
        self.cut_h = self.output_size['height']
        self.crop_params = crop_params

    def trigger_crop(self):
        if len(self.json_item['shapes']) == 1:
            p_list, ori_cluster = [np.array(self.json_item['shapes'][0]['points'])], [np.array([0])]
        else:
            p_list, ori_cluster = self.cluster_points(self.json_item['shapes'])
        crop_proxy = self.get_crop_proxy(p_list, bias=self.crop_params.get('bias', True),
                                         resample=self.crop_params.get('resample', 1),
                                         mega_instance_policy=self.crop_params.get('mega_instance_policy', 'center'))
        new_img_list, new_json_list = self.crop_img(crop_proxy)
        return new_img_list, new_json_list

    def cluster_points(self, shape_list):  # return list of clusters of points and  list of clusters of index
        p_list, ori_cluster = [], []
        for index, item in enumerate(shape_list):
            if item['shape_type'] in ['circle', 'point']:
                center_point = np.array(item['points'][0])
                try:
                    r = np.linalg.norm(np.array(item['points'][1]) - center_point)
                except:
                    r = item.get('width', 6)
                p_list.append(
                    np.array([[center_point[0] - r, center_point[1] - r], [center_point[0] + r, center_point[1] + r]]))
            else:
                p_list.append(np.array(item['points']))
            ori_cluster.append(np.array([index]))
        while True:
            p_new_list, new_cluster = [], []
            center_dict = compute_center(p_list)  # {id: center_point}
            dis_matrix = pairwise_distances(list(center_dict.values()))
            while len(center_dict) > 0:
                if len(center_dict) == 1:
                    new_cluster.append(ori_cluster[list(center_dict.keys())[0]])
                    p_new_list.append(p_list[list(center_dict.keys())[0]])
                    center_dict, dis_matrix = del_point(0, center_dict, dis_matrix)
                    continue
                p_candidate_index = int(np.where(dis_matrix[0] == np.min(dis_matrix[0][1:]))[0])
                cluster_candidate = np.concatenate(
                    (p_list[list(center_dict.keys())[0]], p_list[list(center_dict.keys())[p_candidate_index]])).T
                if np.max(cluster_candidate[0]) - np.min(cluster_candidate[0]) < self.cut_w and np.max(
                        cluster_candidate[1]) - np.min(cluster_candidate[1]) < self.cut_h:
                    new_cluster.append(np.concatenate((ori_cluster[list(center_dict.keys())[0]],
                                                       ori_cluster[list(center_dict.keys())[p_candidate_index]])))
                    p_new_list.append(cluster_candidate.T)
                    center_dict, dis_matrix = del_point(p_candidate_index, center_dict,
                                                        dis_matrix)  # remove the later point at first
                    center_dict, dis_matrix = del_point(0, center_dict, dis_matrix)
                else:
                    new_cluster.append(ori_cluster[list(center_dict.keys())[0]])
                    p_new_list.append(p_list[list(center_dict.keys())[0]])
                    center_dict, dis_matrix = del_point(0, center_dict, dis_matrix)
            if new_cluster == ori_cluster:
                break
            ori_cluster = new_cluster
            p_list = p_new_list
        return p_list, ori_cluster

    def get_crop_proxy(self, point_clusters, bias, resample, mega_instance_policy):
        """return a list of proxies of crop:
        [ {crop_x, crop_y, ratio}, ... ]
        """
        assert isinstance(bias, bool), 'bias only accept bool'
        assert isinstance(resample, int), 'resample only accept int'
        assert mega_instance_policy in ['center', 'resize',
                                        'cut'], 'mega_instance_policy should in [\'center\', \'resize\', \'cut\']'
        proxy_list = []
        img_h, img_w = self.img_item.shape[:2]
        for clus_item in point_clusters:
            clus_item_T = clus_item.T
            max_w, max_h = np.max(clus_item_T[0]) - np.min(clus_item_T[0]), np.max(clus_item_T[1]) - np.min(
                clus_item_T[1])
            resize_ratio = np.max([1.0, max_w / self.cut_w, max_h / self.cut_h])
            crop_x_l, crop_y_l, resize_ratio_l = [], [], []
            if resize_ratio > 1.0:
                if mega_instance_policy == 'resize':
                    mu = random.uniform(0.8, 0.95)
                    resize_ratio_l.append(resize_ratio / mu)  # expand the crop area
                    crop_x_l.append((np.max(clus_item_T[0]) + np.min(clus_item_T[0]) - self.cut_w * mu) / 2)
                    crop_y_l.append((np.max(clus_item_T[1]) + np.min(clus_item_T[1]) - self.cut_h * mu) / 2)
                elif mega_instance_policy == 'cut':
                    x_num, y_num = math.ceil(max_w / self.cut_w), math.ceil(max_h / self.cut_h)
                    for block_i in range(x_num * y_num):
                        x_i, y_i = block_i % x_num, int(block_i / x_num)
                        crop_x_item = np.min(clus_item_T[0]) + (2 * x_i) * max_w / (2 * x_num) - self.cut_w
                        crop_y_item = np.min(clus_item_T[1]) + (2 * y_i) * max_h / (2 * y_num) - self.cut_h
                        crop_x_l.append(crop_x_item)
                        crop_y_l.append(crop_y_item)
                        resize_ratio_l.append(1)
                elif mega_instance_policy == 'center':
                    resize_ratio_l.append(1)
                    crop_x_l.append((np.max(clus_item_T[0]) + np.min(clus_item_T[0]) - self.cut_w) / 2)
                    crop_y_l.append((np.max(clus_item_T[1]) + np.min(clus_item_T[1]) - self.cut_h) / 2)
            else:
                resize_ratio_l.append(1)
                crop_x_l.append((np.max(clus_item_T[0]) + np.min(clus_item_T[0]) - self.cut_w) / 2)
                crop_y_l.append((np.max(clus_item_T[1]) + np.min(clus_item_T[1]) - self.cut_h) / 2)
            for _ in range(resample):
                for crop_x_item, crop_y_item, resize_ratio_item in zip(crop_x_l, crop_y_l, resize_ratio_l):
                    if bias:
                        w_bias = int(random.uniform(-0.15, 0.15) * self.cut_w)
                        h_bias = int(random.uniform(-0.15, 0.15) * self.cut_h)
                        crop_x_item, crop_y_item = crop_x_item + w_bias, crop_y_item + h_bias
                    crop_x_item = judge_loc(crop_x_item, int(self.cut_w * resize_ratio_item), img_w)
                    crop_y_item = judge_loc(crop_y_item, int(self.cut_h * resize_ratio_item), img_h)
                    proxy_list.append({'crop_x': crop_x_item, 'crop_y': crop_y_item, 'ratio': resize_ratio_item})
        return proxy_list

    def crop_img(self, crop_proxy):
        h, w = self.img_item.shape[:2]
        name = Path(self.json_item['imagePath']).stem
        shape_list = self.json_item['shapes']
        mask_poly = []
        for shape_item in shape_list:
            mask_poly.append(lm_shape_to_mask((h, w), shape_item['points'], shape_item['shape_type'],
                                              line_width=int(shape_item.get('width', 5))))
        mask_poly = np.stack(mask_poly, axis=0)
        crop_img_list, crop_json_list = [], []
        for crop_item in crop_proxy:
            img_item = copy.deepcopy(self.img_item)
            mask_poly_instance = copy.deepcopy(mask_poly)
            crop_x, crop_y, resize_ratio = int(crop_item['crop_x']), int(crop_item['crop_y']), crop_item['ratio']
            end_x, end_y = (crop_x + self.cut_w, crop_y + self.cut_h) if resize_ratio == 1.0 \
                else (crop_x + int(self.cut_w * resize_ratio), crop_y + int(self.cut_h * resize_ratio))

            # Pad the img if necessary
            x_pad_f, x_pad_e = np.max([0, 0 - crop_x]), np.max([0, end_x - w])
            y_pad_f, y_pad_e = np.max([0, 0 - crop_y]), np.max([0, end_y - h])
            crop_x, end_x = (crop_x + x_pad_f, end_x + x_pad_f) if x_pad_f > 0 else (crop_x, end_x)
            crop_y, end_y = (crop_y + y_pad_f, end_y + y_pad_f) if y_pad_f > 0 else (crop_y, end_y)
            if np.any(np.array([x_pad_f, x_pad_e, y_pad_f, y_pad_e]) > 0):
                mask_poly_instance = np.pad(mask_poly_instance, ((0, 0), (y_pad_f, y_pad_e), (x_pad_f, x_pad_e)),
                                            'constant', constant_values=0)
                img_item = cv2.copyMakeBorder(img_item, y_pad_f, y_pad_e, x_pad_f, x_pad_e,
                                              cv2.BORDER_CONSTANT, value=(114, 114, 114))

            channel_list = [k for k, g in
                            itertools.groupby(np.where(mask_poly_instance[:, crop_y:end_y, crop_x:end_x] != 0)[0])]
            # get shape list
            obj_item = []
            for shape_index, mask_item in zip(channel_list,
                                              mask_poly_instance[channel_list, crop_y:end_y, crop_x:end_x]):
                mask_item = np.asfortranarray(mask_item).astype('uint8')
                if resize_ratio > 1:
                    mask_item = cv2.resize(mask_item, (self.cut_h, self.cut_w),
                                           interpolation=cv2.INTER_NEAREST)
                contour_extractor = ContourExtractor(mask=mask_item)
                contour_point = contour_extractor.get_contour()
                if not contour_point:
                    continue
                else:
                    for polygon_item in contour_point:
                        shape_copy = copy.deepcopy(shape_list[shape_index])
                        shape_copy['points'] = [polygon_item[i: i + 2] for i in range(0, len(polygon_item), 2)]
                        shape_copy['shape_type'] = 'polygon'
                        obj_item.append(shape_copy)
            json_data = copy.deepcopy(self.json_item)
            data_id = '{}_{}_{}'.format(crop_x, crop_y, name)
            json_data['imagePath'] = data_id + '.jpg'
            img_item = img_item[crop_y: end_y, crop_x: end_x] if resize_ratio == 1.0 \
                else cv2.resize(img_item[crop_y: end_y, crop_x: end_x], (self.cut_h, self.cut_w),
                                interpolation=cv2.INTER_LINEAR)
            json_data['imageHeight'] = self.cut_h
            json_data['imageWidth'] = self.cut_w
            json_data['shapes'] = obj_item
            crop_img_list.append(img_item)
            crop_json_list.append(json_data)
        return crop_img_list, crop_json_list


class RecursiveCrop:
    # support crop_params: None
    def __init__(self, img_item, json_item, output_size, crop_params):
        self.img_item = img_item
        self.json_item = json_item
        self.output_size = output_size
        self.cut_w = self.output_size['width']
        self.cut_h = self.output_size['height']
        self.crop_params = crop_params

    def save_new_img(self, img_np, xmin, ymin, xmax, ymax, img_x, img_y):
        # 切图并保存
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        left, top, right, down = 0, 0, 0, 0  # need padding size
        if xmax > img_x:
            right = xmax - img_x
            xmax = img_x
            # print('out of width')
        if ymax > img_y:
            down = ymax - img_y
            ymax = img_y
            # print('out of hight')
        if ymin < 0:
            top = abs(ymin)
            ymin = 0
            # print('out of hight')
        if xmin < 0:
            left = abs(xmin)
            xmin = 0
            # # print('out of width')
        img_crop = img_np[ymin:ymax, xmin:xmax]
        try:
            ret = cv2.copyMakeBorder(img_crop, top, down, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # padding
            return ret
        except:
            raise
        return None

    def count_bbox_size(self, per_object):
        points = per_object['points']
        x, y = zip(*points)  # split x,y
        if per_object['shape_type'] == 'circle':
            center_point = points[0]
            r_p = points[1]
            r = round(math.sqrt((center_point[0] - r_p[0]) ** 2 + (center_point[1] - r_p[1]) ** 2), 2)
            min_x = round(center_point[0] - r, 2)
            min_y = round(center_point[1] - r, 2)
            max_x = round(center_point[0] + r, 2)
            max_y = round(center_point[1] + r, 2)
        else:
            min_x = round(min(x), 2)
            min_y = round(min(y), 2)
            max_x = round(max(x), 2)
            max_y = round(max(y), 2)
        # print('max_x,max_y,min_x,min_y',max_x,max_y,min_x,min_y,'---',i['shape_type'])
        return max_x, max_y, min_x, min_y

    def get_new_location(self, point, mid_point, crop_w=64, crop_h=64):
        # 将缺陷放于中心位置
        p_x = point[0] - mid_point[0] + crop_w / 2
        p_y = point[1] - mid_point[1] + crop_h / 2
        if p_x < 0:
            p_x = 0
        if p_y < 0:
            p_y = 0
        if p_x > crop_w:
            p_x = crop_w
        if p_y > crop_h:
            p_y = crop_h
        return [p_x, p_y]

    def trigger_crop(self):
        img_np = self.img_item  # 原图数据
        shapes_img_l = {}
        c = 0
        for i in self.json_item['shapes']:
            c += 1
            shapes_img_l[c] = i
        cut_one_img = []
        mid_point = []
        resize_ratio = []
        data_id = os.path.splitext(os.path.basename(self.json_item['imagePath']))[0]
        self.recursion_cut(shapes_img_l, self.cut_w, self.cut_h, cut_one_img, mid_point, resize_ratio)  # 聚类
        new_img_list = []
        new_json_list = []
        for index_object in range(len(cut_one_img)):
            bias_w = self.cut_w / 2 - mid_point[index_object][0] if mid_point[index_object][
                                                                        0] - self.cut_w / 2 < 0 else 0
            bias_h = self.cut_h / 2 - mid_point[index_object][1] if mid_point[index_object][
                                                                        1] - self.cut_h / 2 < 0 else 0
            bias_w = self.img_item.shape[1] - mid_point[index_object][0] - self.cut_w / 2 if mid_point[index_object][
                                                                                                 0] + self.cut_w / 2 > \
                                                                                             self.img_item.shape[
                                                                                                 1] else bias_w
            bias_h = self.img_item.shape[0] - mid_point[index_object][1] - self.cut_h / 2 if mid_point[index_object][
                                                                                                 1] + self.cut_h / 2 > \
                                                                                             self.img_item.shape[
                                                                                                 0] else bias_h
            for shapes_object in cut_one_img[index_object]:
                new_points = []
                for loc in shapes_object['points']:
                    n_p = self.get_new_location(loc, mid_point[index_object], self.cut_w, self.cut_h)
                    n_p[0] -= bias_w
                    n_p[1] -= bias_h
                    new_points.append(n_p)
                shapes_object['points'] = new_points
            new_name_img = '{}_{}_{}.jpg'.format(mid_point[index_object][0], mid_point[index_object][1],
                                                 data_id)
            # 生成新的img文件，抠图过程中会出现超出边界的坐标
            source_x_min, source_x_max = mid_point[index_object][0] - self.cut_w / 2 + bias_w, mid_point[index_object][
                0] + self.cut_w / 2 + bias_w  # 抠图位置
            source_y_min, source_y_max = mid_point[index_object][1] - self.cut_h / 2 + bias_h, mid_point[index_object][
                1] + self.cut_h / 2 + bias_h
            x_min, x_max, y_min, y_max = int(source_x_min), int(source_x_max), int(source_y_min), int(source_y_max)
            # new_name_img
            target_size = (math.ceil(self.img_item.shape[1] * resize_ratio[index_object]),
                           math.ceil(self.img_item.shape[0] * resize_ratio[index_object]))
            new_img = self.save_new_img(cv2.resize(img_np, target_size, interpolation=cv2.INTER_LINEAR), x_min, y_min,
                                        x_max, y_max, self.img_item.shape[1] * resize_ratio[index_object],
                                        self.img_item.shape[0] * resize_ratio[index_object])
            # 生成新的json文件
            # crop_szie_w,crop_szie_h = crop_szie,crop_szie
            new_json = self.def_new_json(self.json_item, self.cut_w, self.cut_h, cut_one_img[index_object],
                                         new_name_img)
            if new_img is not None and new_json:
                new_img_list.append(new_img)
                new_json_list.append(new_json)
            else:
                print('Failed to crop the {}'.format(cut_one_img[index_object]))
        return new_img_list, new_json_list

    def def_new_json(self, json_data, crop_szie_w, crop_size_h, shapes_img, new_name_img):
        new_json = {}
        # new_json['flags'] = json_data['flags']
        new_json['imageData'] = None
        # new_json['imageDepth'] = json_data['imageDepth']
        new_json['imageHeight'] = crop_size_h
        # new_json['imageLabeled'] = json_data['imageLabeled']
        new_json['imagePath'] = new_name_img
        new_json['imageWidth'] = crop_szie_w
        new_json['shapes'] = shapes_img
        # new_json['time_Labeled'] = json_data['time_Labeled']
        new_json['version'] = json_data['version']
        return new_json

    def recursion_cut(self, shapes_img_l, crop_w, crop_h, cut_one_img, mid_point, resize_ratio):
        if len(shapes_img_l) == 0:
            # print('递归结束了',counter_per_cut)
            return 0
        next_allow = {}  # 记录不可以放一起的标注
        allow = []
        max_bbox = []
        for i in shapes_img_l:
            max_x, max_y, min_x, min_y = self.count_bbox_size(shapes_img_l[i])  # 获取标注的位置
            # process the situation of mega instance
            if max_x - min_x > crop_w or max_y - min_y > crop_h:
                try:
                    if self.crop_params.get('mega_instance_policy', 'center') == 'center':
                        img_shape = (self.img_item.shape[0], self.img_item.shape[1])
                        contour_extractor = ContourExtractor(shape_item=shapes_img_l[i], img_shape=img_shape)
                        center_point = (round((min_x + max_x) / 2), round((min_y + max_y) / 2))
                        start_x, start_y = max((center_point[0] - round(crop_w / 2)), 0), max(
                            (center_point[1] - round(crop_h / 2)), 0)
                        start_x, start_y = min(self.img_item.shape[1], start_x + crop_w) - (crop_w), min(
                            self.img_item.shape[0], start_y + crop_h) - (crop_h)
                        crop_loc = {'start_x': start_x, 'end_x': start_x + crop_w, 'start_y': start_y,
                                    'end_y': start_y + crop_h}
                        contour_point = contour_extractor.get_contour(crop_loc=crop_loc)
                        polygon_list = []
                        for polygon_item in contour_point:
                            shape_copy = copy.deepcopy(shapes_img_l[i])
                            shape_copy['points'] = [[polygon_item[i] + start_x, polygon_item[i + 1] + start_y] for i in
                                                    range(0, len(polygon_item), 2)]
                            shape_copy['shape_type'] = 'polygon'
                            polygon_list.append(shape_copy)
                        cut_one_img.append(polygon_list)
                        mid_point.append(
                            (math.ceil(crop_loc['start_x'] + crop_w / 2), math.ceil(crop_loc['start_y'] + crop_h / 2)))
                        resize_ratio.append(1)
                        continue
                    elif self.crop_params['mega_instance_policy'] == 'resize':
                        img_shape = (self.img_item.shape[0], self.img_item.shape[1])
                        contour_extractor = ContourExtractor(shape_item=shapes_img_l[i], img_shape=img_shape)
                        c_ratio = min(crop_w / (max_x - min_x), crop_h / (max_y - min_y)) * random.randint(80, 95) / 100
                        center_point = (round((min_x + max_x) * c_ratio / 2), round((min_y + max_y) * c_ratio / 2))
                        target_size = (
                            math.ceil(self.img_item.shape[1] * c_ratio), math.ceil(self.img_item.shape[0] * c_ratio))
                        contour_extractor.mask_binary = cv2.resize(contour_extractor.mask_binary, target_size,
                                                                   interpolation=cv2.INTER_LINEAR)
                        start_x, start_y = max((center_point[0] - round(crop_w / 2)), 0), max(
                            (center_point[1] - round(crop_h / 2)), 0)
                        start_x, start_y = min(target_size[0], start_x + crop_w) - (crop_w), min(target_size[1],
                                                                                                 start_y + crop_h) - (
                                               crop_h)
                        crop_loc = {'start_x': start_x, 'end_x': start_x + crop_w, 'start_y': start_y,
                                    'end_y': start_y + crop_h}
                        contour_point = contour_extractor.get_contour(crop_loc=crop_loc)
                        polygon_list = []
                        for polygon_item in contour_point:
                            shape_copy = copy.deepcopy(shapes_img_l[i])
                            shape_copy['points'] = [[polygon_item[i] + start_x, polygon_item[i + 1] + start_y] for i in
                                                    range(0, len(polygon_item), 2)]
                            shape_copy['shape_type'] = 'polygon'
                            polygon_list.append(shape_copy)
                        cut_one_img.append(polygon_list)
                        mid_point.append(
                            (math.ceil(crop_loc['start_x'] + crop_w / 2), math.ceil(crop_loc['start_y'] + crop_h / 2)))
                        resize_ratio.append(c_ratio)
                        continue
                except KeyError as e:
                    raise e
            # 与已有点比较距离
            if len(max_bbox) > 0:
                a, b, c, d = max_bbox
                mmin_x = min(min_x, c)
                mmin_y = min(min_y, d)
                mmax_x = max(max_x, a)
                mmax_y = max(max_y, b)
                ww, hh = mmax_x - mmin_x, mmax_y - mmin_y
                # print('最大长宽',ww,hh)
                if ww < crop_w and hh < crop_h:
                    max_bbox = mmax_x, mmax_y, mmin_x, mmin_y
                    allow.append(shapes_img_l[i])
                else:
                    next_allow[i] = shapes_img_l[i]  # 不可以放一起的
            else:
                max_bbox = [max_x, max_y, min_x, min_y]
                allow.append(shapes_img_l[i])

        # 计算聚类后类别在原图的中心点。
        if allow:
            w, h = max_bbox[0] - max_bbox[2], max_bbox[1] - max_bbox[3]
            mid_x = math.ceil(max_bbox[2] + w / 2)
            mid_y = math.ceil(max_bbox[3] + h / 2)
            # print('中心点',math.ceil(mid_x),math.ceil(mid_y))
            cut_one_img.append(allow)
            mid_point.append((mid_x, mid_y))
            resize_ratio.append(1)
        self.recursion_cut(next_allow, crop_w, crop_h, cut_one_img, mid_point, resize_ratio)


class GridCrop:
    # support crop_params: None
    def __init__(self, img_item, json_item, output_size, crop_params):
        self.img_item = img_item
        self.json_item = json_item
        self.output_size = output_size
        self.cut_w = self.output_size['width']
        self.cut_h = self.output_size['height']
        self.crop_params = crop_params

    def trigger_crop(self):
        nums_cut_w = self.crop_params.get('num_w')
        nums_cut_h = self.crop_params.get('num_h')
        if not nums_cut_w or not nums_cut_h:
            raise KeyError('Params \'num_w\' and \'num_h\' are required for grid_crop.')
        overlap = self.crop_params.get('overlap') if self.crop_params.get('overlap') else 0
        return self.crop_image(nums_cut_w, nums_cut_h, overlap)

    def crop_image(self, nums_cut_w, nums_cut_h, overlap):
        name = os.path.splitext(os.path.split(self.json_item['imagePath'])[-1])[0]
        json_data = copy.deepcopy(self.json_item)
        overlap_w = overlap * self.img_item.shape[1]
        overlap_h = overlap * self.img_item.shape[0]
        crop_width = self.img_item.shape[1] // nums_cut_w + overlap_w
        crop_height = self.img_item.shape[0] // nums_cut_h + overlap_h
        height, width = self.img_item.shape[0], self.img_item.shape[1]
        start_x, start_y = 0, 0
        num_index = 0
        img_slice_dict = {}
        img_list, json_list = [], []

        for y_index in range(nums_cut_h):
            end_y = min(start_y + crop_height, height)
            if end_y - start_y < crop_height:
                start_y = end_y - crop_height
            for x_index in range(nums_cut_w):
                num_index += 1
                end_x = min(start_x + crop_width, width)
                if end_x - start_x < crop_width:
                    start_x = end_x - crop_width
                img_slice_dict[y_index * nums_cut_w + x_index] = {'start_x': start_x, 'end_x': end_x,
                                                                  'start_y': start_y, 'end_y': end_y}
                start_x = start_x + crop_width - overlap_w
            start_x = 0
            start_y = start_y + crop_height - overlap_h

        shapes_json_data = json_data['shapes']
        object_dict = {}
        for shape_object in shapes_json_data:
            img_shape = (self.img_item.shape[0], self.img_item.shape[1])
            contour_extractor = ContourExtractor(shape_item=shape_object, img_shape=img_shape)
            slice_idx = 0
            while slice_idx < len(img_slice_dict):
                contour_point = contour_extractor.get_contour(crop_loc=img_slice_dict[slice_idx])
                if not contour_point:
                    slice_idx += 1
                    continue
                else:
                    for polygon_item in contour_point:
                        shape_copy = copy.deepcopy(shape_object)
                        shape_copy['points'] = [polygon_item[i: i + 2] for i in range(0, len(polygon_item), 2)]
                        shape_copy['shape_type'] = 'polygon'
                        try:
                            object_dict[slice_idx].append(shape_copy)
                        except:
                            object_dict[slice_idx] = [shape_copy]
                    slice_idx += 1

        for obj_idx, obj_item in object_dict.items():
            json_data = copy.deepcopy(self.json_item)
            data_id = '{}_{}_{}'.format(obj_idx % nums_cut_w, obj_idx // nums_cut_w, name)
            json_data['imagePath'] = data_id + '.jpg'
            img_item = self.img_item[img_slice_dict[obj_idx]['start_y']: img_slice_dict[obj_idx]['end_y'],
                       img_slice_dict[obj_idx]['start_x']: img_slice_dict[obj_idx]['end_x']]
            json_data['imageHeight'] = img_item.shape[0]
            json_data['imageWidth'] = img_item.shape[1]
            json_data['shapes'] = obj_item
            img_list.append(img_item)
            json_list.append(json_data)

        return img_list, json_list
