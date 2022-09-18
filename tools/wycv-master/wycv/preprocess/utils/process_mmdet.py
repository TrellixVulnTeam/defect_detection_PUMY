'''
@Description: 生成测试集的annotation用于对img和类别进行说明，并且切为指定大小的图
@author: lijianqing
@date: 2020/12/11 16:44
@return
'''
# ---------------------------class1----------------------start
# 统计jsons中的类别数量
# [('guashang', 15070), ('maoxu', 14505), ('keli', 11499), ('heidian', 11140)]
# ['guashang', 'maoxu', 'keli', 'heidian']
# 4
# 目标汇总数： 52214

# 调用说明：if __name__ == '__main__':
#           a = Counter_cate(r'D:\work\data\microsoft\jalama\data\train\dm\13141516\cutdm\jsons')
# 输入路径为存放json文件的路径，若有图像或其他格式文件存在时并不影响统计。
import json
import glob


class Counter_cate(object):
    def __init__(self, jsonpath):
        self.counter_cate(jsonpath)  #

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def counter_cate(self, json_path):
        jsons = glob.glob(r'{}\*.json'.format(json_path))
        dic = {}
        for i in jsons:
            ret_dic = self.parse_para(i)
            shapes = ret_dic['shapes']
            for j in shapes:
                if j['label'] not in dic:
                    dic[j['label']] = 1
                else:
                    dic[j['label']] += 1
        a = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        print(a)
        b = [i[0] for i in a]
        print(b)
        print(len(a))
        sum = 0
        for i in dic:
            sum += dic[i]
        print('目标汇总数：', sum)
        return list(dic.keys())


# ---------------------------class1----------------------end


# ---------------------------class4----------------------start
# 使用随机采样方式划分数据集
# 输入参数：数据列表路径，labelme路径，保存路径，列表数据后缀，（保存路径不存在时会自动创建）,ratio划分比例
# 调用：labels_path = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\tt\val20171'
# labelme_path = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\tt\val2017'
# save_path = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\tt\1'
# RandomSplitDataset(labels_path,labelme_path,save_path,'png',ratio=0.3)
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil


class RandomSplitDataset(object):
    def __init__(self, labels_path, labelme_path, save_path, flag, ratio=0.2):
        name_ist = os.listdir(labels_path)
        train_set, test_set = train_test_split(name_ist, test_size=ratio, random_state=0)
        print(len(train_set), len(test_set))
        print(train_set)
        print(test_set)
        train_save_p = os.path.join(save_path, 'train2017')
        test_save_p = os.path.join(save_path, 'val2017')
        train_save_stuff_p = os.path.join(save_path, 'stufftrain2017')
        test_save_stuff_p = os.path.join(save_path, 'stuffval2017')

        self.move_data(train_set, labelme_path, train_save_p, flag, labels_path, train_save_stuff_p)
        self.move_data(test_set, labelme_path, test_save_p, flag, labels_path, test_save_stuff_p)

    def move_data(self, data_names, labelme_path, save_path, flag, labels_path, labels_save_p):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(labels_save_p):
            os.makedirs(labels_save_p)
        for i in data_names:
            if i.endswith(flag):
                name_img = i.replace(flag, 'jpg')
                name_json = i.replace(flag, 'json')
                pp_img = os.path.join(labelme_path, name_img)
                ss_img = os.path.join(save_path, name_img)
                pp_json = os.path.join(labelme_path, name_json)
                ss_json = os.path.join(save_path, name_json)
                pp_png = os.path.join(labels_path, i)
                ss_png = os.path.join(labels_save_p, i)
                shutil.move(pp_img, ss_img)
                shutil.move(pp_json, ss_json)
                shutil.move(pp_png, ss_png)


# ---------------------------class4----------------------end


# ---------------------------class5----------------------start
# 对齐coco类别，输入参照coco,待对齐coco,保存coco
import json


class Modify_COCO_Cate(object):
    def __init__(self, cz_coco, coco, save_coco):
        self.cz_coco = cz_coco  # 参照cocojson
        self.coco = coco  # 待修改的cocojson
        self.save_coco = save_coco  # 修改后的cocojson
        self.modify(cz_coco, coco, save_coco)

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)  # indent=4 更加美观显示

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def modify(self, cz_json, coco_json, save_coco_json):
        cz_json_data = self.parse_para(cz_json)
        cz_categories = cz_json_data['categories']
        coco_json_data = self.parse_para(coco_json)
        coco_json_cate = coco_json_data['categories']
        save_coco_dic = {}
        cz_cate_dic = {}
        coco_id_2_id_cate_dic = {}
        for i in cz_categories:
            cate_name = i['supercategory']
            cate_id = i['id']
            cz_cate_dic[cate_name] = cate_id
        for i in coco_json_cate:
            coco_cate_id = i['id']
            coco_cate_name = i['supercategory']
            coco_id_2_id_cate_dic[coco_cate_id] = cz_cate_dic[coco_cate_name]

        coco_annotations = coco_json_data['annotations']
        save_coco_annotations = []
        for i in coco_annotations:
            i['category_id'] = coco_id_2_id_cate_dic[i['category_id']]
            save_coco_annotations.append(i)

        save_coco_dic['images'] = coco_json_data['images']
        save_coco_dic['categories'] = cz_categories
        save_coco_dic['annotations'] = save_coco_annotations
        self.save_json(save_coco_dic, save_coco_json)


# ---------------------------class5----------------------end

# ---------------------------class6----------------------start
# 对于coco格式的文件进行框抑制。输入coco格式的路径，需要保存的路径，以及抑制iou阈值。输出一个类，调用输出路径。nms = Pre_nms('C:/Users/xie5817026/PycharmProjects/pythonProject1/1228/htc462.json','./',0.1)
# print(nms.out_coco)
import shutil
import json
import os
import numpy as np


class Pre_nms(object):
    def __init__(self, pre_coco_json, out_p, iou_thr=0.1):  # seg,coco,output
        self.out_coco = self.main(pre_coco_json, out_p, iou_thr=iou_thr)

    def get_box(self, object1):
        xmin, ymin, w, h = object1['bbox']
        xmax, ymax = xmin + w, ymin + h
        return [xmin, ymin, xmax, ymax]

    def get_box_score(self, object1):
        xmin, ymin, w, h = object1['bbox']
        xmax, ymax = xmin + w, ymin + h
        score = object1['score']
        return [xmin, ymin, xmax, ymax, score]

    def save_json(self, dic, path):
        json.dump(dic, open(path, 'w', encoding='utf-8'), indent=4)
        return 0

    def parse_para(self, input_json):  # 解析标注数据
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
            images = ret_dic['images']
            categories = ret_dic['categories']
            annotations = ret_dic['annotations']
        dic_img_id = {}
        for i_obj in annotations:
            img_id = i_obj['image_id']
            if img_id in dic_img_id:
                dic_img_id[img_id].append(i_obj)
            else:
                dic_img_id[img_id] = [i_obj]
        print('dic_img_id:', len(dic_img_id))
        return (images, categories, dic_img_id)

    def compute_iou(self, bbox1, bbox2):
        """
        compute iou
        :param bbox1:
        :param bbox2:
        :return: iou
        """
        bbox1xmin = bbox1[0]
        bbox1ymin = bbox1[1]
        bbox1xmax = bbox1[2]
        bbox1ymax = bbox1[3]
        bbox2xmin = bbox2[0]
        bbox2ymin = bbox2[1]
        bbox2xmax = bbox2[2]
        bbox2ymax = bbox2[3]
        area1 = (bbox1ymax - bbox1ymin) * (bbox1xmax - bbox1xmin)
        area2 = (bbox2ymax - bbox2ymin) * (bbox2xmax - bbox2xmin)
        bboxxmin = max(bbox1xmin, bbox2xmin)
        bboxxmax = min(bbox1xmax, bbox2xmax)
        bboxymin = max(bbox1ymin, bbox2ymin)
        bboxymax = min(bbox1ymax, bbox2ymax)
        if bboxxmin >= bboxxmax:
            return 0
        if bboxymin >= bboxymax:
            return 0
        area = (bboxymax - bboxymin) * (bboxxmax - bboxxmin)
        iou = area / (area1 + area2 - area)
        return iou

    def single_img_annotation(self, annotations):  # 预测结果json,annotations=[{'image_id':1,...},{}]，flag=1预测标注；flag=0实际标注
        img_id_anno_dic = {}
        print(len(annotations), '--')
        for annotation in annotations:
            try:
                image_id = annotation['image_id']
                # print(annotation,'--',c)
            except:
                print('---1')

            if image_id in img_id_anno_dic:
                img_id_anno_dic[image_id].append(annotation)
            else:
                img_id_anno_dic[image_id] = [annotation]
        return img_id_anno_dic  # {'1':[{},{}]}

    def py_cpu_nms(self, dets, thresh):
        # print('det',dets)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []
        index = scores.argsort()[::-1]
        while index.size > 0:
            i = index[0]  # every time the first is the biggst, and add it directly
            keep.append(i)
            x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]  # because index start from 1
        return keep

    def save_nms_coco_model(self, images, categories, annotations, savejson='pre_nms.json'):
        new_dic = {}
        new_dic['images'] = images
        new_dic['categories'] = categories
        new_dic['annotations'] = annotations
        self.save_json(new_dic, savejson)

    def main(self, coco_json, out_p, iou_thr=0.1):
        out_coco_name = 'htc{}'.format(coco_json.split('htc')[-1])  # instances_test_cemian.json
        # out_coco_name = 'instances{}'.format(coco_json.split('instances')[-1])#instances_test_cemian.json
        print(out_coco_name, '---')
        if not os.path.exists(out_p):
            os.makedirs(out_p)
        out_coco = os.path.join(out_p, out_coco_name)
        images, categories, annotations = self.parse_para(coco_json)  # gt
        annotations_nms_before = []
        annotations_nms_after = []
        for img_id in annotations:  # {'imgid1':[anno1,anno2]}
            anno_img = annotations[img_id]
            if len(anno_img) == 0:
                print(img_id, '预测目标为0')
            boxxes = []
            box_id_dic = {}
            for anno_obj in anno_img:  # [anno1,anno2]
                bbox = self.get_box_score(anno_obj)
                boxxes.append(bbox)
                box_id_dic[len(boxxes) - 1] = anno_obj
                annotations_nms_before.append(anno_obj)
            if len(boxxes) == 0:
                print('det is zero')
            else:
                keep = self.py_cpu_nms(np.array(boxxes), thresh=iou_thr)  # 抑制后的索引
                for index_s in keep:
                    annotations_nms_after.append(box_id_dic[index_s])
        print('iou阈值：{}；抑制前：{}，抑制后：{},rate:{}'.format(iou_thr, len(annotations_nms_before), len(annotations_nms_after),
                                                      len(annotations_nms_after) / len(annotations_nms_before)))
        self.save_nms_coco_model(images, categories, annotations_nms_after, out_coco)
        # for i in annotations_nms_after:
        #     img_i = i['image_id']
        #     print('====',img_i)
        return out_coco


# ---------------------------class6----------------------end
# ---------------------------class7----------------------start
# 实物csv和实物图可视化，给他实物图和实物csv生成对应的xml标注和标注合并图，显示的时候注意类别映射字典。调用：#csv_p = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\ProductGradeMaterialCheck.csv'
# img_p ='D:\work\data\microsoft\jalama\data\heduiji\merge_all",ShiwuHedui(img_p,csv_p),生成'D:\work\data\microsoft\jalama\data\heduiji\merge_all\outputs",'D:\work\data\microsoft\jalama\data\heduiji\merge_all\r_imgs"
import cv2
import os
import glob
import pandas as ps
import shutil
from PIL import Image
# @Description:
# @Author     : zhangyan
# @Time       : 2021/1/14 3:54 下午

import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom


class Dic2xml(object):
    def __init__(self, dic, xml_save_path):
        self.dic2xml(dic, xml_save_path)

    def create_Node(self, element, text=None):
        elem = ET.Element(element)
        elem.text = text
        return elem

    def link_Node(self, root, element, text=None):
        """
        @param root: element的父节点
        @param element: 创建的element子节点
        @param text: element节点内容
        @return: 创建的子节点
        """
        if text != None:
            text = str(text)
        element = self.create_Node(element, text)
        root.append(element)
        return element

    # 保存为XML文件（美化后）
    def saveXML(self, root, filename, indent="\t", newl="\n", encoding="utf-8"):
        rawText = ET.tostring(root)
        dom = minidom.parseString(rawText)
        with open(filename, 'w', encoding="utf-8") as f:
            dom.writexml(f, "", indent, newl, encoding)

    def get_dic_data(self, key, value):
        save_name = key.split('.')[0] + '.xml'
        anno = value.get('anno')
        w = value.get('w')
        h = value.get('h')
        return save_name, save_name, anno, None, 'true', w, h, 3

    def generate_xml(self, key, value, xml_save_path):
        save_name, xmlpath, anno, time_label, image_label, width, height, depth = self.get_dic_data(key, value)

        root = ET.Element("doc")  # 创建根结点

        path = self.link_Node(root, 'path', xmlpath)  # 创建path节点
        outputs = self.link_Node(root, 'outputs')
        object = self.link_Node(outputs, 'object')

        for i in range(len(anno)):
            item = self.link_Node(object, 'item')  # 创建item节点

            label = anno[i][4]  # 获取label
            width_points_line = 2  # 点或线的width
            shape_type = 'rectangle'

            name = self.link_Node(item, 'name', label)  # 添加json信息到item中
            width_2 = self.link_Node(item, 'width', width_points_line)

            if shape_type == 'rectangle':
                bndbox = self.link_Node(item, 'bndbox')
                xmin = self.link_Node(bndbox, 'xmin', int(anno[i][0]))
                ymin = self.link_Node(bndbox, 'ymin', int(anno[i][1]))
                xmax = self.link_Node(bndbox, 'xmax', int(anno[i][2]))
                ymax = self.link_Node(bndbox, 'ymax', int(anno[i][3]))

            status = self.link_Node(item, 'status', str(1))

        time_labeled = self.link_Node(root, 'time_labeled', time_label)  # 创建time_labeled节点
        labeled = self.link_Node(root, 'labeled', image_label)
        size = self.link_Node(root, 'size')
        width = self.link_Node(size, 'width', width)
        height = self.link_Node(size, 'height', height)
        depth = self.link_Node(size, 'depth', depth)

        save_path = os.path.join(xml_save_path, save_name)
        if not os.path.exists(xml_save_path):
            os.makedirs(xml_save_path)
        # 保存xml文件
        self.saveXML(root, save_path)
        print('{}'.format(save_name) + ' has been transformed!')

    def dic2xml(self, dic, xml_save_path):
        t = time.time()
        for key, value in dic.items():
            self.generate_xml(key, value, xml_save_path)
        print(time.time() - t)


# if __name__ == '__main__':
#     dic = {'123.jpg': {'anno': [(132, 243, 355, 467, '刮伤'), (51, 61, 72, 82, '异色')], 'w': 512, 'h': 512},
#            '456.jpg': {'anno':[(11, 21, 31, 41, '擦伤'), (11, 22, 33, 41, '白点')], 'w': 512, 'h': 512}}
#
#     xml_save_path = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\xml'
#     Dic2xml(dic, xml_save_path)
class ShiwuHedui(object):
    def __init__(self, source_p, csv_p):
        # 图像位置
        # csv_p = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\ProductGradeMaterialCheck.csv'
        dic_wh = self.w_h(source_p)  # 图像存放位置
        img_dic_c = self.read_csv(csv_p, dic_wh)  # csv格式
        xml_save_path = os.path.join(source_p,
                                     'outputs')  # r'D:\work\data\microsoft\jalama\data\heduiji\merge_all\outputs'#xml保存位置，没有的时候会自动创建
        save_p = os.path.join(source_p,
                              'r_img')  # r'D:\work\data\microsoft\jalama\data\heduiji\merge_all\r_img'#标注和原图合并的位置，有则保存，无则创建再保存。
        # 任务号,工件号,图号,缺陷,PointX,PointY,Width,Height,PhysicalExpression
        # 127,30,15,8,4921,754,83,57,明显
        # source_p = r'D:\work\data\microsoft\jalama\data\heduiji\merge_all'
        if not os.path.exists(save_p):
            os.makedirs(save_p)
        if not os.path.exists(xml_save_path):
            os.makedirs(xml_save_path)
        Dic2xml(img_dic_c, xml_save_path)
        self.iter_dic(source_p, save_p, img_dic_c)  # 原始图像位置，保存位置，图像标注逻辑数据。

    def w_h(self, p):
        dic_wh = {}
        print('---')
        for i in os.listdir(p):
            i_p = os.path.join(p, i)
            # data = cv2.imread(i_p)
            data = Image.open(i_p)
            dic_wh[i] = data.size
            print(data.size)
        return dic_wh

    def mv_all(self, path_root, save):
        for i in os.listdir(path_root):
            f_p = os.path.join(path_root, i)
            for j in os.listdir(f_p):
                s_p = os.path.join(f_p, j)
                save_p = os.path.join(save, j)
                shutil.copy(s_p, save_p)

    # mv_all(r'D:\work\data\microsoft\jalama\data\heduiji\heduiji',r'D:\work\data\microsoft\jalama\data\heduiji\merge_all')
    def keys_l(self, task_id, l):
        task_ids = []
        for i in task_id:
            # print(i,'----')
            if len(str(i)) != l:
                sy = l - len(str(i))
                if sy != 0:
                    ii = i
                    for j in range(sy):
                        ii = '0' + str(ii)
                    task_ids.append(ii)

                else:
                    task_ids.append(str(i))
            else:
                task_ids.append(str(i))
        return task_ids

    def read_csv(self, csv_path, dic_wh):
        dic_qx_name = {0: '良品', 1: '异色', 2: '白点', 4: '刮伤', 6: '黑点', 7: '砂痕印', 8: '异物',
                       9: '刀纹', 10: '刮伤', 12: '应力痕', 13: '凸凹痕', 14: '凹凸痕',
                       15: '凹坑', 19: '碰伤', 21: '21', 22: '刀纹线', 23: '塌边', 24: '颗粒',
                       25: '毛絮', 26: '线痕', 27: '掉漆', 28: '变形', 29: '加铣',
                       30: '过切', 33: '凹凸痕1', 34: '凹凸痕2'}  # 图像和名字对应表
        r = ps.read_csv(csv_path, usecols=['任务号', '工件号', '图号', '缺陷', 'PointX', 'PointY', 'Width', 'Height'])
        imgs_dic = {}
        task_id = r['任务号']
        task_ids = self.keys_l(task_id, 4)
        gongjian_id = r['工件号']
        gongjian_ids = self.keys_l(gongjian_id, 4)
        img_id = r['图号']
        img_ids = self.keys_l(img_id, 2)
        x_min = r['PointX']
        y_min = r['PointY']
        w = r['Width']
        h = r['Height']
        label = r['缺陷']

        for i in range(len(img_id)):
            img_name = '{}-{}-{}.jpg'.format(task_ids[i], gongjian_ids[i], img_ids[i])
            x_max = x_min[i] + w[i]
            y_max = y_min[i] + h[i]
            if img_name in imgs_dic:
                imgs_dic[img_name].append((x_min[i], y_min[i], x_max, y_max, dic_qx_name[label[i]]))
            else:
                imgs_dic[img_name] = [(x_min[i], y_min[i], x_max, y_max, dic_qx_name[label[i]])]
        img_dic_c = {}
        for i in imgs_dic:
            try:
                w, h = dic_wh[i]
                one_img_dic = {}
                one_img_dic['anno'] = imgs_dic[i]
                one_img_dic['w'] = w
                one_img_dic['h'] = h
                img_dic_c[i] = one_img_dic
            except:
                print('--')
        print(img_dic_c)
        return img_dic_c

    def draw_bbox(self, img, left_top, right_down, color, label):
        cv2.rectangle(img, left_top, right_down, color, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, left_top, font, 1, color, 1)
        return img

    def iter_dic(self, source_p, save_p, dic_img_c):
        for i in os.listdir(source_p):
            if i.endswith('.jpg'):
                img_n = os.path.join(source_p, i)
                img = cv2.imread(img_n)
                if i in dic_img_c:
                    anno_i = dic_img_c[i]['anno']
                    print('anno', anno_i)
                    for j in anno_i:
                        print('jjj', j)
                        l_t = (int(j[0]), int(j[1]))
                        r_d = (int(j[2]), int(j[3]))
                        color = (0, 192, 255)
                        label = j[4]
                        img = self.draw_bbox(img, l_t, r_d, color, label)
                cv2.imwrite(os.path.join(save_p, i), img)


# ---------------------------class7----------------------end
# ---------------------------class8----------------------start
# 拆分每个光学面的数据到单独的文件夹
import os
import shutil


# o_p = r'D:\work\data\microsoft\jalama\data\train\d\0114大面'#输入路径
# o_p_1 =r'D:\work\data\microsoft\jalama\data\train\d\0120大面'#保存路径# r'D:\work\data\microsoft\damian\damian_source\1027data\classfile\ds\gsyshd\x512cut\刮伤黑点'#p#r'D:\work\data\microsoft\damian\damian_source\1027data\classfile\凹凸痕382'
# 调用：Split_channel(o_p,o_p_1)
class Split_channel(object):
    def __init__(self, o_p, o_p_1):
        a = 0
        for i in os.listdir(o_p):
            if i.endswith('.jpg'):
                name = i.split('.jpg')[0]
                channel_id = name.split('-')[-1]
                print('channel_id', channel_id)
                try:
                    channel_path = os.path.join(o_p_1, channel_id)
                    if not os.path.exists(channel_path):
                        os.makedirs(channel_path)
                    outputs_path = os.path.join(channel_path, 'outputs')
                    source_path = os.path.join(o_p, 'outputs')
                    if not os.path.exists(outputs_path):
                        os.makedirs(outputs_path)
                    shutil.move(os.path.join(o_p, i), os.path.join(channel_path, i))  # img
                    xml_name = i.replace('jpg', 'xml')
                    xml_path = os.path.join(outputs_path, xml_name)
                    source_xml_path = os.path.join(source_path, xml_name)
                    shutil.move(source_xml_path, xml_path)  # img
                except:
                    print(0)


# ---------------------------class8----------------------end
# ---------------------------class9----------------------start
# 以某个文件夹的列表名为准，移动另一个文件夹的内容到指定文件夹，ls_p：列表名路径,flag：列表文件后缀,m_flag：移动文件后缀,s_p：待移动文件路径,save_p#保存路径，调用：
# Mv_l('ls_p',’png‘,'jpg','a','b')
import os
import shutil
import glob


class Mv_l(object):
    def __init__(self, ls_p, flag, m_flag, s_p, save_p):
        if not os.path.exists(save_p):
            os.makedirs(save_p)
        for i in os.listdir(ls_p):
            if i.endswith(flag):
                name = i.replace(flag, m_flag)
                print('name:', name)
                shutil.copy(os.path.join(s_p, name), os.path.join(save_p, name))


# ---------------------------class9----------------------end
# ---------------------------class10----------------------start
# 筛选包含某些指定类别的数据，移动到另一个文件下，同时移动img,json,调用： 数据路径，保存路径，类别列表，labels = ['heidian']，source_p，save_p =r'D:\work\data\microsoft\a\train\dmwuy\dm_heidian'
# SelectLabel(source_p,labels,save_p)
import json
import os
import shutil


class SelectLabel(object):
    def __init__(self, source_p, labels, save_p):
        print('input----')
        self.get_img_ls(source_p, labels, save_p)

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            img_id_lables = {}
            ret_dic = json.load(f)
            shapes = ret_dic['shapes']
            imagePath = ret_dic['imagePath']
            for i in shapes:
                label = i['label']
                if imagePath in img_id_lables:
                    img_id_lables[imagePath].append(label)
                else:
                    img_id_lables[imagePath] = [label]
        return (imagePath, img_id_lables)

    def mv_dic(self, dic, source_path, save_path):
        for img_id in dic:
            json_name = img_id.replace('jpg', 'json')
            try:
                print('正在移动：', json_name)
                shutil.move(os.path.join(source_path, img_id), os.path.join(save_path, img_id))
                shutil.move(os.path.join(source_path, json_name), os.path.join(save_path, json_name))
            except:
                print(img_id, 'or', json_name, '移动错误')

    def get_img_ls(self, labelme_jsons, labels, save_path):
        level_one = {}  # 只包含指定类别的图像，给定【1，2，3，4】，结果【1，2，3，4】，【1，2，2，3，4】
        level_two = {}  # 只包含部分指定类别的图像，给定【1，2，3，4】，结果【1，2，3】，【2，3，4】
        level_three = {}  # 只包含部分指定类别的图像，给定【1，2，3，4】，结果【1，2，3，5】，【2，3，4，7】
        c = 0
        for i in os.listdir(labelme_jsons):
            # print('--labelme_jsons',labelme_jsons)
            if i.endswith('.json'):
                c += 1
                print('i', c, '--', i)
                json_path = os.path.join(labelme_jsons, i)
                img_name, label_ls = self.parse_para(json_path)
                intersection_set = set(label_ls[img_name]).intersection(set(labels))
                if len(intersection_set):
                    if len(set(label_ls[img_name])) == len(labels) and len(intersection_set) == len(labels):
                        # print('len(categories_l)', len(categories_l),'---',len(set(img_id_dic[i])))
                        level_one[img_name] = len(intersection_set)
                    elif len(set(label_ls[img_name])) < len(labels) and set(label_ls[img_name]).issubset(set(labels)):
                        level_two[img_name] = len(intersection_set)
                        # print(img_id_dic[i],'--------',list(set(img_id_dic[i])),'--------',list(intersection_set), '-------', categories_l)
                    else:
                        level_three[img_name] = len(intersection_set)
        file_name_ls = ['all', 'part', 'others']
        file_dic_ls = [level_one, level_two, level_three]
        for i in range(len(file_name_ls)):
            s_p = os.path.join(save_path, file_name_ls[i])
            if not os.path.exists(s_p):
                os.makedirs(s_p)
            self.mv_dic(file_dic_ls[i], labelme_jsons, s_p)


# ---------------------------class10----------------------end


import cv2
import os
import math
import json


class GenerateCutTestCoco(object):
    def __init__(self, root_path, cz_json, cut_flag=False, overlap_w=0, overlap_h=0, img_save_path='./',
                 coco_save_path='./intances_test2017.json'):
        self.root_path = root_path
        self.overlap_w = overlap_w
        self.overlap_h = overlap_h
        self.img_save_path = img_save_path
        self.coco_save_path = coco_save_path
        self.cut_flag = cut_flag
        self.cz_json = cz_json
        self.num_index = 0
        self.main()

    def image(self, data, num):
        image = {}
        height, width = data["imageHeight"], data["imageWidth"]
        image['height'] = height
        image['width'] = width
        image['id'] = num
        image['file_name'] = data['imagePath'].split('/')[-1]
        return image

    def save_new_img(self, img_np, img_name, xmin, ymin, xmax, ymax, out_path, source_img_w, source_img_h):
        # 切图并保存
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        left, top, right, down = 0, 0, 0, 0  # need padding size
        if xmax > source_img_w:
            down = xmax - source_img_w
            xmax = source_img_w
            # print('out of width,xmax',xmax)
        if ymax > source_img_h:
            right = ymax - source_img_h
            ymax = source_img_h
            # print('out of hight ymax',ymax)
        if ymin < 0:
            top = abs(ymin)
            ymin = 0
            # print('out of hight')
        if xmin < 0:
            left = abs(xmin)
            xmin = 0
            # # print('out of width')
        img_crop = img_np[ymin:ymax, xmin:xmax]
        ret = cv2.copyMakeBorder(img_crop, top, down, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # padding
        print('os.path.join(out_path, img_name)', os.path.join(out_path, img_name))
        cv2.imwrite(os.path.join(out_path, img_name), ret)

    def crop_image(self, image, nums_cut_w, nums_cut_h, overlab, out_path, name):
        self.crop_width = image.shape[1] // nums_cut_w + overlab
        self.crop_height = image.shape[0] // nums_cut_h + overlab
        self.overlap = overlab
        height, width = image.shape[0], image.shape[1]
        height_len = (height - self.overlap) // (self.crop_height - self.overlap) + 1
        width_len = (width - self.overlap) // (self.crop_width - self.overlap) + 1
        start_x, start_y = 0, 0
        imgs_annotation = []
        for y_index in range(height_len):
            end_y = min(start_y + self.crop_height, height)
            if end_y - start_y < self.crop_height:
                start_y = end_y - self.crop_height
            for x_index in range(width_len):
                self.num_index += 1
                end_x = min(start_x + self.crop_width, width)
                if end_x - start_x < self.crop_width:
                    start_x = end_x - self.crop_width
                img_crop = image[start_y: end_y, start_x: end_x]
                # img_crop = image[start_y: end_y + 1, start_x: end_x + 1]
                crop_img_name = '{}_{}_{}'.format(x_index, y_index, name)
                data = {'imageWidth': self.crop_width, 'imageHeight': self.crop_height, 'imagePath': crop_img_name}
                imgs_annotation.append(self.image(data, self.num_index))
                # ret = cv2.copyMakeBorder(img_crop, top, down, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))#padding
                print('os.path.join(out_path, img_name)', os.path.join(out_path, crop_img_name))
                cv2.imwrite(os.path.join(out_path, crop_img_name), img_crop)
                start_x = start_x + self.crop_width - self.overlap
            start_x = 0
            start_y = start_y + self.crop_height - self.overlap
        return imgs_annotation

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)

    def main(self):
        imgs_annotation = []
        if self.cut_flag:
            for i in os.listdir(self.root_path):
                img_np = cv2.imread(os.path.join(self.root_path, i))  # 原图数据
                img_annotation = self.crop_image(img_np, 2, 2, 50, self.img_save_path, i)
                imgs_annotation.extend(img_annotation)
        else:
            num_index = 0
            for i in os.listdir(self.root_path):
                num_index += 1
                img_np = cv2.imread(os.path.join(self.root_path, i))
                img_h, img_w, _ = img_np.shape
                data = {'imageWidth': img_w, 'imageHeight': img_h, 'imagePath': i}
                imgs_annotation.append(self.image(data, num_index))
        json_data = self.parse_para(self.cz_json)
        test_dic = {}
        test_dic['images'] = imgs_annotation
        test_dic['categories'] = json_data['categories']
        test_dic['annotations'] = []
        self.save_json(test_dic, self.coco_save_path)


'''
@Description: TODO
@author: lijianqing
@date: 2020/12/14 16:39
@return
'''
import json


class MergeTestResult2coco(object):
    def __init__(self, test_result_json, test_coco_json, save_path='./260_test1225.json'):
        self.test_result_json = test_result_json
        self.test_coco_json = test_coco_json
        self.save_path = save_path
        coco_data = self.parse_para(test_coco_json)
        coco_data['annotations'] = self.parse_para(test_result_json)
        self.save_json(coco_data, self.save_path)

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)


from pycocotools.mask import decode
import numpy as np
import json
import cv2
import os

'''
@Description: 将coco格式的json转化为对应图像的json文件，用于labelme查看或后续生成可使用的coco
@author: lijianqing
@date: 2020/12/11 15:37
'''


class Coco2Labelme(object):
    def __init__(self, coco_path, out_path, score):
        self.coco_path = coco_path
        self.out_path = out_path
        self.score = score
        self.main()

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)

    def def_new_json(self, i, new_name, out_p):
        new_json = {}
        new_json['flags'] = {}
        new_json['imageData'] = None
        new_json['imageDepth'] = 3
        new_json['imageHeight'] = i['img_h']
        new_json['imageLabeled'] = "true"
        new_json['imagePath'] = i['img_path']
        new_json['imageWidth'] = i['img_w']
        new_json['shapes'] = i['shapes']
        new_json['time_Labeled'] = None
        new_json['version'] = "1.0"
        self.save_json(new_json, os.path.join(out_p, new_name))
        # print('生成了',os.path.join(out_p,new_name))
        return new_json

    def def_dic_element(self, shapes_img, i):
        dic_element = {}
        dic_element['label'] = i['label']
        # shape_type = modify_type('polygon',i['points'])
        dic_element['width'] = 1
        if len(i['points']) == 1:
            points = [[i['bbox'][0], i['bbox'][1]], [i['bbox'][0] + i['bbox'][2], i['bbox'][1] + i['bbox'][3]]]
            shape_type = 'rectangle'
        else:
            points = i['points']
            shape_type = 'polygon'
        dic_element['shape_type'] = shape_type
        dic_element['points'] = points
        dic_element['group_id'] = ""
        dic_element['flags'] = {}
        dic_element['level'] = ""
        # dic_element['area']=i['area']
        dic_element['bbox'] = i['bbox']
        dic_element['score'] = i['score']
        dic_element['imagePathSource'] = i['imagePathSource']
        shapes_img.append(dic_element)
        return shapes_img

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def parse_img_c_a(self, coco_data):
        imgs_cate = coco_data['images']
        cate_cate = coco_data['categories']
        annotations = coco_data['annotations']
        img_id_name_dic = {}
        cate_id_name_dic = {}
        annotation_imgid_anno_dic = {}
        for i in imgs_cate:
            img_id_name_dic[i['id']] = i['file_name']
        for i in cate_cate:
            cate_id_name_dic[i['id']] = i['name']
        for i in annotations:
            img_id = i['image_id']
            if img_id not in annotation_imgid_anno_dic:
                annotation_imgid_anno_dic[img_id] = [i]
            else:
                annotation_imgid_anno_dic[img_id].append(i)
        return (img_id_name_dic, cate_id_name_dic, annotation_imgid_anno_dic)

    # mask转二值图 黑白两色
    def mask2bw(self, mask):
        # print('mask_shape',mask)
        # print(mask)
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask[i][j] == 1:
                    mask[i][j] = 255
        return mask

    # 提取二值图轮廓
    def getContoursBinary(self, blimg):
        print(blimg.shape)
        ret, binary = cv2.threshold(blimg, 0.5, 255, cv2.THRESH_BINARY)
        print(binary.shape)
        # ret, binary = cv2.threshold(blimg, 127, 255, cv2.THRESH_BINARY)
        # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        return contours

    def main(self):
        print('self.coco_path', self.coco_path)
        coco_data = self.parse_para(self.coco_path)
        img_id_name_dic, cate_id_name_dic, annotation_imgid_anno_dic = self.parse_img_c_a(coco_data)
        for i in annotation_imgid_anno_dic:
            shapes = []
            one_img_anno = annotation_imgid_anno_dic[i]
            obj = {}
            anno_dic = {}
            for j in one_img_anno:
                if j['score'] >= self.score:
                    segmentation = j['segmentation']
                    img_w, img_h = segmentation['size']
                    mask = decode(segmentation)
                    contours = self.getContoursBinary(mask)
                    if len(contours) != 0:
                        points = np.squeeze(contours[0]).tolist()
                        if len(np.shape(points)) == 1:  # (1,1,2)-np.squeeze->(2,)  ---->(1,2)
                            points = [points]
                        obj['label'] = cate_id_name_dic[j['category_id']]
                        # obj['area']=j['area']
                        obj['bbox'] = j['bbox']
                        obj['score'] = j['score']
                        obj['imagePathSource'] = ""
                        obj['points'] = points
                        print('score', obj['score'])
                        shapes = self.def_dic_element(shapes, obj)
                        print('--', shapes)
            #
            anno_dic['shapes'] = shapes
            anno_dic['img_w'] = img_w
            anno_dic['img_h'] = img_h
            # print('img_id_name_dic',img_id_name_dic)
            anno_dic['img_path'] = img_id_name_dic[i]
            new_name = img_id_name_dic[i].replace('.jpg', '.json')
            self.def_new_json(anno_dic, new_name, self.out_path)


import os
import json
import cv2


class Merge_cut4(object):
    def __init__(self, cut_json_p, source_img_p, save_p):
        self.main(cut_json_p, source_img_p, save_p)

    def parse_para(self, input_json):
        print('input_json', input_json)
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)

    def def_new_json(self, new_name, out_p, w, h, shapes):
        new_json = {}
        new_json['flags'] = {}
        new_json['imageData'] = None
        new_json['imageDepth'] = 3
        new_json['imageHeight'] = h
        new_json['imageLabeled'] = "true"
        new_json['imagePath'] = new_name
        new_json['imageWidth'] = w
        new_json['shapes'] = shapes
        new_json['time_Labeled'] = None
        new_json['version'] = "1.0"
        self.save_json(new_json, out_p)
        # print('生成了',os.path.join(out_p,new_name))
        return new_json

    def fy(self, x, y, points, w, h):
        x0 = 0
        y0 = 0
        # print('jjj',y)
        if x == '1':
            x0 = w / 2 - 50
        if y == '1':
            y0 = h / 2 - 50
            print('y0', y0)
        points_new = []
        for i in points:
            x = i[0] + x0
            y = i[1] + y0
            points_new.append([x, y])
        return points_new

    def main(self, cut_json_p, source_img_p, save_p):
        # cut_json_p = r'D:\work\data\microsoft\jalama\test1231\1203\1203damian\0.1\nms\275'#切图的json文件，不可以带与图像混合存放
        # source_img_p = r'D:\work\data\microsoft\jalama\test1231\1203\1203damian\imgs'#原图的图像文件
        # save_p = r'D:\work\data\microsoft\jalama\test1231\1203\1203damian\0.1merge\275'#合并图的结果文件
        img_dic = {}
        for n in os.listdir(cut_json_p):
            print('n', n)
            x, y, name = n.split('_')
            img_name = name.replace('.json', '.jpg')
            if img_name in img_dic:
                img_dic[img_name].append(n)
            else:
                img_dic[img_name] = [n]
        print('len', len(img_dic))
        # print('len',img_dic)
        for img_one in img_dic:
            img_p = os.path.join(source_img_p, img_one)
            imgsize = cv2.imread(img_p).shape
            h = imgsize[0]
            w = imgsize[1]
            shapes = []
            # print(h,w)
            for json_one in img_dic[img_one]:
                j_p = os.path.join(cut_json_p, json_one)
                print(j_p)
                data = self.parse_para(j_p)
                shapes_json = data['shapes']
                for k in shapes_json:
                    points = k['points']
                    x, y, name = json_one.split('_')
                    print(x, y, '--')
                    new_points = self.fy(x, y, points, w, h)
                    k['points'] = new_points
                    shapes.append(k)
            json_name = img_one.replace('.jpg', '.json')
            save_name = os.path.join(save_p, json_name)
            # print(save_name)
            self.def_new_json(new_name=img_one, out_p=save_name, w=w, h=h, shapes=shapes)


import numpy as np
import json
import pandas as pd
import itertools
import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import shutil


class AnnalyResult(object):
    def __init__(self, yt_labelme, test_labelme, out_path, title_png):
        self.yt_labelme = yt_labelme
        self.test_labelme = test_labelme
        self.out_path = out_path
        self.title_png = title_png
        self.gt_class = []
        self.pre_class = []
        start_time = time.time()
        self.main()
        end_time = time.time()
        print('run time:', end_time - start_time)
        self.compute_confmx()

    def get_points_box(self, points, type='polygon', width=2):
        points = np.array(points)
        if type == 'point' and len(points) == 1:
            box = [points[0][0] - width / 2, points[0][1] - width / 2, points[0][0] + width / 2,
                   points[0][1] + width / 2]
            return box
        if type == 'circle' and len(points) == 2:
            r = np.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2)
            box = [points[0][0] - r, points[0][1] - r, points[0][0] + r, points[0][1] + r]
            return box
        box = [min(points[:, 0]), min(points[:, 1]), max(points[:, 0]), max(points[:, 1])]
        return box

    def parse_para_re(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def save_json(self, dic, path):
        json.dump(dic, open(path, 'w', encoding='utf-8'), indent=4)
        return 0

    def compute_iou(self, bbox1, bbox2):
        """
        compute iou
        :param bbox1:
        :param bbox2:
        :return: iou
        """
        bbox1xmin = bbox1[0]
        bbox1ymin = bbox1[1]
        bbox1xmax = bbox1[2]
        bbox1ymax = bbox1[3]
        bbox2xmin = bbox2[0]
        bbox2ymin = bbox2[1]
        bbox2xmax = bbox2[2]
        bbox2ymax = bbox2[3]
        area1 = (bbox1ymax - bbox1ymin) * (bbox1xmax - bbox1xmin)
        area2 = (bbox2ymax - bbox2ymin) * (bbox2xmax - bbox2xmin)
        bboxxmin = max(bbox1xmin, bbox2xmin)
        bboxxmax = min(bbox1xmax, bbox2xmax)
        bboxymin = max(bbox1ymin, bbox2ymin)
        bboxymax = min(bbox1ymax, bbox2ymax)
        if bboxxmin >= bboxxmax:
            return 0
        if bboxymin >= bboxymax:
            return 0
        area = (bboxymax - bboxymin) * (bboxxmax - bboxxmin)
        iou = area / (area1 + area2 - area)
        return iou

    def l_g_ls(self, require, part, iou_thr, f_l_flag):  # 过检记录：某个预测框与所有gt的iou等于0，表明该预测框为过检框,一张图预测不出结果时需要考虑漏检如何计算。
        result_l = []
        for j in require:
            result_flag = 0
            gt_points = j['points']
            for k in part:
                pre_points = k['points']
                bbox_gt = self.get_points_box(gt_points, j['shape_type'])
                bbox_re = self.get_points_box(pre_points, k['shape_type'])
                iou = self.compute_iou(bbox_gt, bbox_re)
                if iou < iou_thr:  # <iou_thr:
                    # print('iou:',iou),#过检和漏检与gt的iou都为0
                    result_flag += 1
                    # print('r',result_flag,len(part))
            if result_flag == len(part):  # iou为0的数量与所有预测标注的数量是否相等，若相等表明缺陷漏检，若为0的记录小于0则表明缺陷未漏检。
                if f_l_flag == 'loujian':  # loushi
                    self.gt_class.append(j['label'])
                    self.pre_class.append('z_lou_or_guo')
                else:  # guojian
                    self.gt_class.append('z_lou_or_guo')
                    self.pre_class.append(j['label'])
                result_l.append(j)
                print('result_l', result_l)
        return result_l

    def jiandui_ls(self, require, part, iou_thr):
        jd = []
        for j in require:
            gt_points = j['points']
            for k in part:
                pre_points = k['points']
                bbox_gt = self.get_points_box(gt_points, j['shape_type'])
                bbox_re = self.get_points_box(pre_points, k['shape_type'])
                iou = self.compute_iou(bbox_gt, bbox_re)
                if iou >= iou_thr:
                    if not k in jd:
                        jd.append(k)
                        self.gt_class.append(j['label'])
                        self.pre_class.append(k['label'])
        return jd

    def compute_confmx(self):
        classes = sorted(list(set(self.gt_class)), reverse=False)  # 类别排序
        cm = confusion_matrix(self.gt_class, self.pre_class, classes)  # 根据类别生成矩阵，此处不需要转置
        cm_pro = (cm.T / np.sum(cm, 1)).T
        print('cm', cm)
        print('cmp', cm_pro)

        self.plot_confusion_matrix(cm, classes, 'nums')
        self.plot_confusion_matrix(cm_pro, classes, 'pro', normalize=True)
        print('confx', cm)

    def new_json(self, cz, shapes, save_json):
        new_json_dic = {}
        new_json_dic['flags'] = cz['flags']
        new_json_dic['imageData'] = cz['imageData']
        new_json_dic['imageDepth'] = cz['imageDepth']
        new_json_dic['imageLabeled'] = cz['imageLabeled']
        new_json_dic['imagePath'] = cz['imagePath']
        new_json_dic['imageHeight'] = cz['imageHeight']
        new_json_dic['imageWidth'] = cz['imageWidth']
        new_json_dic['shapes'] = shapes
        new_json_dic['time_Labeled'] = cz['time_Labeled']
        new_json_dic['version'] = cz['version']
        if len(shapes) != 0:
            self.save_json(new_json_dic, save_json)

    def proce_compute(self, input_json, pre_json, save_path):
        gt_anno_data = self.parse_para_re(input_json)
        print('gt_json', input_json)
        pre_anno_data = self.parse_para_re(pre_json)
        gt_shapes = gt_anno_data['shapes']
        pre_shapes = pre_anno_data['shapes']
        jiandui_shapes = []
        jiandui_shapes = self.jiandui_ls(gt_shapes, pre_shapes, 0.01)
        guojian_shapes = []
        guojian_shapes = self.l_g_ls(pre_shapes, gt_shapes, 0.01, 'guojian')
        merge_gt_pre_shapes = []
        merge_gt_pre_shapes.extend(gt_shapes)
        merge_gt_pre_shapes.extend(guojian_shapes)
        loujian_shapes = []
        try:
            loujian_shapes = self.l_g_ls(gt_shapes, pre_shapes, 0.01, 'loujian')
        except:
            loujian_shapes.extend(gt_shapes)
        print('---', len(guojian_shapes), len(loujian_shapes), len(jiandui_shapes), len(gt_shapes),
              len(merge_gt_pre_shapes))
        guojian_path = os.path.join(save_path, 'guojian')
        loujian_path = os.path.join(save_path, 'loujian')
        jiandui_path = os.path.join(save_path, 'jiandui')
        merge_gt_pre_path = os.path.join(save_path, 'merge_gt_pre')
        if not os.path.exists(guojian_path):
            os.makedirs(guojian_path)
        if not os.path.exists(loujian_path):
            os.makedirs(loujian_path)
        if not os.path.exists(jiandui_path):
            os.makedirs(jiandui_path)
        if not os.path.exists(merge_gt_pre_path):
            os.makedirs(merge_gt_pre_path)
        img_name = gt_anno_data['imagePath']
        json_name = img_name.replace('.jpg', '.json')
        guojian_json = os.path.join(guojian_path, json_name)
        loujian_json = os.path.join(loujian_path, json_name)
        jiandui_json = os.path.join(jiandui_path, json_name)
        merge_gt_pre_json = os.path.join(merge_gt_pre_path, json_name)
        self.new_json(gt_anno_data, guojian_shapes, guojian_json)
        self.new_json(gt_anno_data, loujian_shapes, loujian_json)
        self.new_json(gt_anno_data, jiandui_shapes, jiandui_json)
        self.new_json(gt_anno_data, merge_gt_pre_shapes, merge_gt_pre_json)

    def main(self):
        for i in os.listdir(self.yt_labelme):
            if i.endswith('.json'):
                input_json = os.path.join(self.yt_labelme, i)
                pre_json = os.path.join(self.test_labelme, i)
                except_json = os.path.join("D:/work/data/microsoft/jalama/sixth/third_cut/test/exception", i)
                try:
                    self.proce_compute(input_json, pre_json, self.out_path)
                except:
                    shutil.move(input_json, except_json)
                    print('未预测数据', input_json)

    def plot_confusion_matrix(self, cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        # plt.figure()

        plt.figure(figsize=(12, 8), dpi=120)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('{}_{}'.format(self.title_png, title))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        # plt.axis("equal")
        ax = plt.gca()
        left, right = plt.xlim()
        ax.spines['left'].set_position(('data', left))
        ax.spines['right'].set_position(('data', right))
        for edge_i in ['top', 'bottom', 'right', 'left']:
            ax.spines[edge_i].set_edgecolor("white")

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            num = float('{:.2f}'.format(cm[i, j])) if normalize else int(cm[i, j])
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")
        plt.ylabel('ground turth')
        plt.xlabel('predict')
        plt.tight_layout()
        save_p = os.path.join(self.out_path, './{}_{}.png'.format(self.title_png, title))
        cm_txt = save_p.replace('.png', '.txt')
        with open(cm_txt, 'a+') as f:
            f.write('{}:\n'.format(title))
            f.write(str(cm))
            f.write('\n')
        # plt.savefig(save_p, transparent=True, dpi=800)
        plt.savefig(save_p, transparent=True, dpi=300)
        # plt.show()


# @Description: json->xml
# @Author     : zhangyan
# @Time       : 2020/12/25 4:28 下午

import json
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from concurrent.futures.thread import ThreadPoolExecutor


class Json2Xml(object):
    def __init__(self, json_path_file, xml_save_path):
        print(0)
        # category = {'a': '良品', 'aotuhen': '凹凸痕', 'aotuhen1': '凹凸痕1', 'aotuhen2': '凹凸痕2', 'baidian': '白点', 'bianxing': '变形',
        #             'daowen': '刀纹', 'diaoqi': '掉漆', 'guashang': '刮伤', 'guoqie': '过切', 'heidian': '黑点', 'jiaxi': '加铣',
        #             'keli': '颗粒', 'maoxu': '毛絮', 'pengshang': '碰伤', 'tabian': '塌边', 'xianhen': '线痕', 'yashang': '压伤',
        #             'yinglihen': '应力痕', 'yise': '异色', 'yiwu': '异物'}
        # category = {'a': '良品_模型', 'aotuhen': '凹凸痕_模型', 'aotuhen1': '凹凸痕1_模型', 'aotuhen2': '凹凸痕2_模型', 'baidian': '白点_模型', 'bianxing': '变形_模型',
        #             'daowen': '刀纹_模型', 'diaoqi': '掉漆_模型', 'guashang': '刮伤_模型', 'guoqie': '过切_模型', 'heidian': '黑点_模型', 'jiaxi': '加铣_模型',
        #             'keli': '颗粒_模型', 'maoxu': '毛絮_模型', 'pengshang': '碰伤_模型', 'tabian': '塌边_模型', 'xianhen': '线痕_模型', 'yashang': '压伤_模型',
        #             'yinglihen': '应力痕_模型', 'yise': '异色_模型', 'yiwu': '异物_模型'}
        category = {'a': '良品', 'aotuhen': '凹凸痕', 'aotuhen1': '凹凸痕1', 'aotuhen2': '凹凸痕2', 'baidian': '白点',
                    'bianxing': '变形',
                    'daowen': '刀纹', 'diaoqi': '掉漆', 'guashang': '刮伤', 'guoqie': '过切', 'heidian': '黑点', 'jiaxi': '加铣',
                    'keli': '颗粒', 'maoxu': '毛絮', 'pengshang': '碰伤', 'tabian': '塌边', 'xianhen': '线痕', 'yashang': '压伤',
                    'yinglihen': '应力痕', 'yise': '异色', 'yiwu': '异物', 'a_moxing': '良品_模型', 'aotuhen_moxing': '凹凸痕_模型',
                    'aotuhen1_moxing': '凹凸痕1_模型', 'aotuhen2_moxing': '凹凸痕2_模型', 'baidian_moxing': '白点_模型',
                    'bianxing_moxing': '变形_模型',
                    'daowen_moxing': '刀纹_模型', 'diaoqi_moxing': '掉漆_模型', 'guashang_moxing': '刮伤_模型',
                    'guoqie_moxing': '过切_模型',
                    'heidian_moxing': '黑点_模型', 'jiaxi_moxing': '加铣_模型',
                    'keli_moxing': '颗粒_模型', 'maoxu_moxing': '毛絮_模型', 'pengshang_moxing': '碰伤_模型',
                    'tabian_moxing': '塌边_模型',
                    'xianhen_moxing': '线痕_模型', 'yashang_moxing': '压伤_模型',
                    'yinglihen_moxing': '应力痕_模型', 'yise_moxing': '异色_模型', 'yiwu_moxing': '异物_模型'}
        # json_path_file = 'D:/work/data/microsoft/jalama/sixth/cuts_modify/cll/jianchu_bz/jsons'
        # xml_save_path = 'D:/work/data/microsoft/jalama/sixth/cuts_modify/cll/jianchu_bz/outputs'
        self.json2xml(json_path_file, xml_save_path, 8, category)

    def create_Node(self, element, text=None):
        elem = ET.Element(element)
        elem.text = text
        return elem

    def link_Node(self, root, element, text=None):
        """
        @param root: element的父节点
        @param element: 创建的element子节点
        @param text: element节点内容
        @return: 创建的子节点
        """
        element = self.create_Node(element, text)
        root.append(element)
        return element

    # 保存为XML文件（美化后）
    def saveXML(self, root, filename, indent="", newl="", encoding="utf-8"):
        try:
            rawText = ET.tostring(root)
            dom = minidom.parseString(rawText)
            with open(filename, 'w', encoding='utf-8') as f:
                dom.writexml(f, "", indent, newl, encoding)
        except:
            # print('----------')
            a = 0

    def json_to_instance(self, json_file_path):
        """
        @param json_file_path: json文件路径
        @return: json_instance
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            instance = json.load(f)
        return instance

    def get_json_data(self, json_path):
        json_data = self.json_to_instance(json_path)

        xmlpath = 'xxx'  # json_data.get('xmlPath')
        img_name = json_data.get('imagePath')
        save_name = img_name.split('.')[0] + '.xml'

        time_label = json_data.get('time_Labeled')  # null
        image_label = json_data.get('imageLabeled')  # true
        width = json_data.get('imageWidth')
        height = json_data.get('imageHeight')
        depth = json_data.get('imageDepth')
        shape = json_data.get('shapes')
        return save_name, xmlpath, shape, time_label, image_label, width, height, depth

    def generate_xml(self, json_path, category, xml_save_path):
        save_name, xmlpath, shape, time_label, image_label, width, height, depth = self.get_json_data(json_path)

        root = ET.Element("doc")  # 创建根结点

        path = self.link_Node(root, 'path', xmlpath)  # 创建path节点
        outputs = self.link_Node(root, 'outputs')
        object = self.link_Node(outputs, 'object')

        for i in range(len(shape)):
            try:
                item = self.link_Node(object, 'item')  # 创建item节点
                shape[i]['flags'] = {}
                label_ori = shape[i].get('label')  # 获取json信息
                label = category.get(label_ori)
                width_points_line = str(shape[i].get('width'))  # 点或线的width
                shape_type = shape[i].get('shape_type')
                points = shape[i].get('points')

                name = self.link_Node(item, 'name', label)  # 添加json信息到item中
                width_2 = self.link_Node(item, 'width', width_points_line)

                if shape_type == 'linestrip':
                    line = self.link_Node(item, 'line')
                    for j in range(len(points)):
                        x = self.link_Node(line, 'x{}'.format(j + 1), str(int(points[j][0])))
                        y = self.link_Node(line, 'y{}'.format(j + 1), str(int(points[j][1])))

                if shape_type == 'polygon':
                    polygon = self.link_Node(item, 'polygon')
                    for j in range(len(points)):
                        x = self.link_Node(polygon, 'x{}'.format(j + 1), str(int(points[j][0])))
                        y = self.link_Node(polygon, 'y{}'.format(j + 1), str(int(points[j][1])))

                if shape_type == 'circle':
                    if int(points[1][0] - points[0][0]) * 2 == width_points_line:
                        point = self.link_Node(item, 'point')
                        x = self.link_Node(point, 'x', str(int(points[0][0])))
                        y = self.link_Node(point, 'y', str(int(points[0][1])))
                    else:
                        ellipse = self.link_Node(item, 'ellipse')
                        xmin = self.link_Node(ellipse, 'xmin', str(int(points[0][0])))
                        ymin = self.link_Node(ellipse, 'ymin', str(int(points[0][1])))
                        xmax = self.link_Node(ellipse, 'xmax', str(int(points[1][0])))
                        ymax = self.link_Node(ellipse, 'ymax', str(int(points[1][1])))

                if shape_type == 'rectangle':
                    bndbox = self.link_Node(item, 'bndbox')
                    xmin = self.link_Node(bndbox, 'xmin', str(int(points[0][0])))
                    ymin = self.link_Node(bndbox, 'ymin', str(int(points[0][1])))
                    xmax = self.link_Node(bndbox, 'xmax', str(int(points[1][0])))
                    ymax = self.link_Node(bndbox, 'ymax', str(int(points[1][1])))

                status = self.link_Node(item, 'status', str(0))
            except:
                print(save_name + '无缺陷')

        time_labeled = self.link_Node(root, 'time_labeled', time_label)  # 创建time_labeled节点
        labeled = self.link_Node(root, 'labeled', image_label)
        size = self.link_Node(root, 'size')
        width = self.link_Node(size, 'width', str(width))
        height = self.link_Node(size, 'height', str(height))
        depth = self.link_Node(size, 'depth', str(depth))

        print('{}'.format(save_name) + ' has been transformed!')
        save_path = os.path.join(xml_save_path, save_name)
        if not os.path.exists(xml_save_path):
            os.makedirs(xml_save_path)
        self.saveXML(root, save_path)

    def json2xml(self, json_path_file, xml_save_path, num_worker, category):
        t = time.time()
        jsonlist = os.listdir(json_path_file)
        thread_pool = ThreadPoolExecutor(max_workers=num_worker)
        print('Thread Pool is created!')
        for json in jsonlist:
            json_path = os.path.join(json_path_file, json)
            thread_pool.submit(self.generate_xml, json_path, category, xml_save_path)
        thread_pool.shutdown(wait=True)
        print(time.time() - t)


class Test_mmdet(object):
    def __init__(self, functions, test_root_path='./', iou_thr=0.5, anno=False):
        if functions == '0':
            print('直接测试集的生成标注文件coco')
            GenerateCutTestCoco(r'D:\work\data\microsoft\jalama\data\300daowen\imgs',
                                r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\train20172\train128\cz\htc_nms462.json',
                                cut_flag=False, overlap_w=50, overlap_h=50,
                                img_save_path=r'D:\work\data\microsoft\jalama\data\300daowen\imgs',
                                coco_save_path=r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\b\instances_test22.json')
            # '''GenerateCutTestCoco(r'D:\work\data\microsoft\jalama\test1231\testdataset\yt',
            #                     # r'C:\Users\xie5817026\PycharmProjects\pythonProject1\1228\instances_test1228.json',
            #                     # cut_flag=False,overlap_w=50,overlap_h=50,
            #                     # img_save_path=r'D:\work\data\microsoft\jalama\test1231\testdataset\cut',
            #                     coco_save_path=r'D:\work\data\microsoft\jalama\test1231\testdataset\instances_test1231.json')'''
        elif functions == '1':
            print('测试集切图并生成标注文件')
            GenerateCutTestCoco(r'D:\work\data\microsoft\jalama\data\20210129\586damian\cut_edge\imgs',
                                r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\instances_train20171.json',
                                cut_flag=True, overlap_w=0, overlap_h=0,
                                img_save_path=r'D:\work\data\microsoft\jalama\data\20210129\586damian\cut_edge\cut',
                                coco_save_path=r'D:\work\data\microsoft\jalama\data\20210129\586damian\instances_test1231.json')
        elif functions == '2':
            print('seg转labelme')
            result_seg_path = os.path.join(test_root_path, 'D:\work\data\microsoft\jalama\data\noanno\coco_result')
            # 先抑制再合并
            MergeTestResult2coco(r'D:\work\data\microsoft\a\0922Atest_cut\test_result\320_dm_cut.segm.json',
                                 r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\annotations\instances_test2017dm.json',
                                 r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\mergeanno\instances_test2017dm_cut.json')
            # nms
            merge_coco = r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\mergeanno\htc320dm.json'
            coco_model_json_nms_path = r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\nms'
            iou_thr = 0.4
            Pre_nms(merge_coco, coco_model_json_nms_path, iou_thr)

            # coco2labelme
            coco2labelme = Coco2Labelme(r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\nms\htc320dm.json',
                                        r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\pre', 0.1)
        elif functions == '3':
            print('合并切图')
            Merge_cut4(cut_json_p=r'D:\work\data\microsoft\a\0922Atest_cut\ANNO\pre',
                       source_img_p=r'D:\work\data\microsoft\a\0922Atest\dm\imgs',
                       save_p=r'D:\work\data\microsoft\a\0922Atest\dm\pre')  ##切图的json文件，不可以带与图像混合存放,
            ##原图的图像文件,#合并图的结果文件
        elif functions == '4':
            print('分析标注结果生成混淆矩阵')
            AnnalyResult(r'D:\work\data\microsoft\a\0922Atest\dm\jsons',  # biaozhu jsons
                         r'D:\work\data\microsoft\a\0922Atest\dm\pre',  # pre jsons
                         r'D:\work\data\microsoft\a\0922Atest\dm\result\320',  # splite result file
                         '320model_0922testdata')  # 混淆矩阵图像名字，不带后缀
        elif functions == '5':  # json2xml
            Json2Xml(json_path_file='D:/work/data/microsoft/jalama/sixth/cuts_modify/cll/jianchu_bz/jsons'
                     , xml_save_path='D:/work/data/microsoft/jalama/sixth/cuts_modify/cll/jianchu_bz/outputs')
        elif functions == '6':
            print('给定无标注测试集，flie:damian,cemian,guaijiao 生成可以用来测试的coco文件')
            test_root_path = test_root_path  # r'D:\work\data\microsoft\jalama\data\noanno'
            cz_path = os.path.join(test_root_path, 'cz')
            coco_path = os.path.join(test_root_path, 'coco')
            coco_result_path = os.path.join(test_root_path, 'coco_result')
            coco_seg_path = os.path.join(coco_result_path, 'seg')
            annotations_path = os.path.join(coco_path, 'annotations')

            if not os.path.exists(annotations_path):
                os.makedirs(annotations_path)
            if not os.path.exists(coco_seg_path):
                os.makedirs(coco_seg_path)
            for i in os.listdir(cz_path):
                cz_json = os.path.join(cz_path, i)
                if not os.path.exists(cz_json):
                    print('没有参照类别的json文件，放入json文件到cz文件夹下。')
            for i in os.listdir(test_root_path):
                imgs_path = os.path.join(test_root_path, i)
                save_imgs_path = os.path.join(coco_path, 'test_{}'.format(i))
                save_json_path = os.path.join(annotations_path, 'instances_test_{}.json'.format(i))
                if i == 'damian':
                    cut_flag = True
                    if not os.path.exists(save_imgs_path):
                        os.makedirs(save_imgs_path)
                    GenerateCutTestCoco(imgs_path, cz_json, cut_flag=cut_flag, overlap_w=50, overlap_h=50,
                                        img_save_path=save_imgs_path, coco_save_path=save_json_path)
                elif i == 'guaijiao' or i == 'cemian':
                    cut_flag = False
                    shutil.copytree(os.path.join(test_root_path, i), save_imgs_path)
                    GenerateCutTestCoco(imgs_path, cz_json, cut_flag=cut_flag, coco_save_path=save_json_path)
                else:
                    a = 0
                    continue
        elif functions == '7':
            print('seg结果生成prejson')
            test_root_path = test_root_path  # r'D:\work\data\microsoft\jalama\data\noanno'
            coco_result_path = os.path.join(test_root_path, 'coco_result')
            analysis_result_path = os.path.join(test_root_path, 'analysis_result')
            coco_seg_path = os.path.join(coco_result_path, 'seg')
            coco_model_json_path = os.path.join(coco_result_path, 'model_json')
            coco_model_json_nms_path = os.path.join(coco_result_path, 'model_nms_json')
            pre_labelme_json_path = os.path.join(coco_result_path, 'pre_labelme')
            coco_path = os.path.join(test_root_path, 'coco')
            annotations_path = os.path.join(coco_path, 'annotations')
            # iou_thr = 0.8
            if not os.path.exists(coco_model_json_path):
                os.makedirs(coco_model_json_path)
            if not os.path.exists(coco_model_json_nms_path):
                os.makedirs(coco_model_json_nms_path)
            for i in os.listdir(coco_seg_path):
                pre_json = os.path.join(coco_seg_path, i)
                print(i, '--')
                model_coco_name = ''
                if i.endswith('damian.segm.json'):
                    model_coco_name = 'instances_test_damian.json'
                    coco_json = os.path.join(annotations_path, model_coco_name)
                    l_name = 'damian_cut'
                elif i.endswith('cemian.segm.json'):
                    model_coco_name = 'instances_test_cemian.json'
                    coco_json = os.path.join(annotations_path, model_coco_name)
                    l_name = 'cemian'
                elif i.endswith('guaijiao.segm.json'):
                    model_coco_name = 'instances_test_guaijiao.json'
                    coco_json = os.path.join(annotations_path, model_coco_name)
                    l_name = 'guaijiao'
                else:
                    continue
                seg_json_path = os.path.join(coco_seg_path, i)
                merge_coco = os.path.join(coco_model_json_path, model_coco_name)
                # 合并seg和coco imgs,categies为pre_coco.json
                MergeTestResult2coco(seg_json_path, coco_json, merge_coco)
                # 对pre_coco.json进行nms得到pre_coco_nms.json
                pre_nms = Pre_nms(merge_coco, coco_model_json_nms_path, iou_thr)
                pre_lableme_name_f = os.path.join(pre_labelme_json_path, l_name)
                if not os.path.exists(pre_lableme_name_f):
                    os.makedirs(pre_lableme_name_f)
                # pre_coco_nms.json转labelmejsons
                score = 0.15
                Coco2Labelme(pre_nms.out_coco, pre_lableme_name_f, score)
                if l_name == 'damian_cut':
                    damian_img_root = os.path.join(test_root_path, 'damian')
                    save_damian_img_root = os.path.join(pre_labelme_json_path, 'damian')
                    if not os.path.exists(save_damian_img_root):
                        os.makedirs(save_damian_img_root)
                    # 对转完的大面进行合并
                    Merge_cut4(cut_json_p=pre_lableme_name_f, source_img_p=damian_img_root,
                               save_p=save_damian_img_root)  ##切图的json文件，不可以带与图像混合存放,
            # 分析gt，pre
            jsons_gt_path = os.path.join(test_root_path, 'jsons')
            pre_labelme_json_path
            if os.path.exists(jsons_gt_path):
                for mian in os.listdir(jsons_gt_path):
                    biaozhujsons = os.path.join(jsons_gt_path, mian)
                    prejsons = os.path.join(pre_labelme_json_path, mian)
                    if not os.path.exists(analysis_result_path):
                        os.makedirs(analysis_result_path)
                    AnnalyResult(biaozhujsons,  # biaozhu jsons
                                 prejsons,  # pre jsons
                                 analysis_result_path,  # splite result file
                                 '{}model_testdata'.format(mian))
        elif functions == '8':
            score = 0.15
            pre_nms = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\1228\0.1\instances_test20171.json'
            coco_model_json_nms_path = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\1228\0.1\instances_test20172.json\instances_test20171.json'
            pre_lableme_name_f = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\train20172\train128\coco\test2017'
            # pre_nmss = Pre_nms(pre_nms,coco_model_json_nms_path,0.5)
            Coco2Labelme(coco_model_json_nms_path, pre_lableme_name_f, score)
        elif functions == '9':
            # test_root_path存放json文件的路径
            Counter_cate(test_root_path)
        elif functions == '10':
            # test_root_path存放数据路径，out_path输出数据路径，有则保存，无则先创建后保存。
            out_path = os.path.join(test_root_path, 'cut_edge')
            Cut_edge(test_root_path, out_path, start_point=[96, 46], crop_w=3308, crop_h=4854, anno=anno)
        elif functions == '11':
            # test_root_path存放数据路径，out_path输出数据路径，有则保存，无则先创建后保存。anno为False不切标注
            out_path = os.path.join(test_root_path, 'heng2shu')
            HorizontalToVertical(test_root_path, out_path, anno=anno)
        elif functions == '12':
            # 随机划分数据集
            # labels_path = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\tt\val20171'
            # labelme_path = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\tt\val2017'
            # save_path = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\tt\1'
            RandomSplitDataset(test_root_path, iou_thr, anno, 'png')
        elif functions == '13':
            cz_json = r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\all\htc_nms462.json'
            coco_json = 'D:/work/data/microsoft/jalama/data/train/dm/13141516/selectcut/other/coco/annotations/instances_train.json'
            save_json_path = 'D:/work/data/microsoft/jalama/data/train/dm/13141516/selectcut/other/coco/annotations/instances_train2017.json'
            Modify_COCO_Cate(cz_json, coco_json, save_json_path)
        elif functions == '14':
            nms = Pre_nms(test_root_path, './', 0.1)
            print(nms.out_coco)
        elif functions == '15':
            # 原图，csv
            source_p = test_root_path
            csv_p = ''
            ShiwuHedui(source_p, csv_p)
        elif functions == '16':
            # 拆分不同相机的图到单独文件夹，输入原始路径，保存路径，保存路径不存在时会自动创建
            o_p = test_root_path
            o_p_1 = ''
            Split_channel(o_p, o_p_1)
        elif functions == '17':
            ##以某个文件夹的列表名为准，移动另一个文件夹的内容到指定文件夹，ls_p：列表名路径,flag：列表文件后缀,m_flag：移动文件后缀,s_p：待移动文件路径,save_p#保存路径，调用：
            ls_p = r'D:\work\data\microsoft\jalama\data\20210129\586damian\cut_edge\imgs\dw\200'
            png = 'json'
            jpg = 'json'
            a = r'D:\work\data\microsoft\jalama\data\20210129\586damian\cut_edge\imgs\imgs'
            b = r'D:\work\data\microsoft\jalama\data\20210129\586damian\cut_edge\imgs\dw\100'
            Mv_l(ls_p, png, jpg, a, b)
            # Mv_l(ls_p,flag,m_flag,s_p,save_p)
        elif functions == '18':
            # 筛选数据集,
            source_p, labels, save_p = test_root_path, iou_thr, anno
            SelectLabel(source_p, labels, save_p)
        elif functions == '19':
            # coco2labelme
            coco_path = os.path.join('D:/work/data/microsoft/jalama/data/20210129/586damian', 'instances_train.json')
            coco_path_s = os.path.join('D:/work/data/microsoft/jalama/data/20210129/586damian', 'jsons_labelme')
            Coco2Labelme(coco_path, coco_path_s, '0.1')
        elif functions == '20':
            merge_coco = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\pre\htc240.json'
            coco_model_json_nms_path = r'C:\Users\xie5817026\PycharmProjects\pythonProject1\0104\nms'
            iou_thr = 0.4
            Pre_nms(merge_coco, coco_model_json_nms_path, iou_thr)
        else:
            print('不做任何操作')
        print('--')


if __name__ == '__main__':
    # HorizontalToVertical(r"F:\module_c_remove_tiny20210305\remove_tiny20210305\damian\test\hengtu",
    #                      r"F:\module_c_remove_tiny20210305\remove_tiny20210305\damian\test\shutu")

    # 切四个框
    root_path = r'F:\module_c_remove_tiny20210305\remove_tiny20210305\damian\test\shutu'
    out_path = r'F:\module_c_remove_tiny20210305\remove_tiny20210305\damian\test\cutedge'
    Cut_edge(root_path, out_path, start_point=[150, 0], crop_w=4500, crop_h=6000, anno=True)

    # xml2coco
    # 9统计文件夹下缺陷标注的数量，输入参数为文件夹位置，主要处理文件为json.json格式文件单独存在或同时存在于文件夹下时都可以使用。[('guashang', 15070), ('maoxu', 14505), ('keli', 11499), ('heidian', 11140)]['guashang', 'maoxu', 'keli', 'heidian'] 目标汇总数： 52214
    # test_mmdet = Test_mmdet('9',r'D:\work\data\microsoft\d_yt')
    # #10切除有标注图像的非光学面位置，同时处理图像和标注文件。
    # test_mmdet = Test_mmdet('10',r'D:\work\data\microsoft\jalama\data\20210129\586damian\defects',anno=True)#其他参数指定请到10处修改。
    # 11将横图和标注转为竖图和标注，或者只转图。
    # test_mmdet = Test_mmdet('11',r'D:\work\data\microsoft\jalama\data\20210129\586damian\jsons',anno=True)#其他参数指定请到11处修改
    # 12将数据集随机划分为训练集和验证集，labels_path，labelme_path，save_path，flag,ratio
    # test_mmdet = Test_mmdet('12',r'labels_path','labelme_path','labelme_path','flag','ratio')#其他参数指定请到11处修改
    # 13调整数据集类别
    # test_mmdet = Test_mmdet('13',r'labels_path','labelme_path','labelme_path','flag','ratio')#其他参数指定请到11处修改
    # 14抑制coco格式的bbox.
    # test_mmdet = Test_mmdet('14',r'coco_path','save_path','iou_thr')#其他参数指定请到11处修改
    # 15可视化转化实物核对的数据，生成对应的xml标注和可视化结果图
    # test_mmdet=Test_mmdet('15',source_p,csv_p)
    # 16拆分不同相机的图到单独文件夹，输入原始路径，保存路径，保存路径不存在时会自动创建
    # test_mmdet=Test_mmdet('16',source_p,save_p)
    # 17以某个文件夹的列表名为准，移动另一个文件夹的内容到指定文件夹，ls_p：列表名路径,flag：列表文件后缀,m_flag：移动文件后缀,s_p：待移动文件路径,save_p
    # test_mmdet=Test_mmdet('17',source_p,save_p)
    a = 0
    # 测试图生成测试格式数据
    # test_mmdet = Test_mmdet('7',r'D:\work\data\microsoft\jalama\data\noanno',0.5)
    # test_mmdet = Test_mmdet('7',r'D:\work\data\microsoft\jalama\data\noanno')
    # test_mmdet = Test_mmdet('8',r'D:\work\data\microsoft\jalama\data\anno\test0114')
    # test_mmdet = Test_mmdet('8',r'D:\work\data\microsoft\jalama\data\anno\test0114')
    # test_mmdet = Test_mmdet('3',r'D:\work\data\microsoft\jalama\data\anno\test0114')
    # test_mmdet = Test_mmdet('0',r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\train20172\train128')
    # test_mmdet = Test_mmdet('6',r'D:\work\data\microsoft\jalama\data\train\dm\13141516\selectcut\other\coco\train20172\train128')
    # 测试结果生成自动标注文件
    # 自动标注与标注对比
    # cut
    # test_mmdet=Test_mmdet('1','','')
    # test_mmdet=Test_mmdet('2','','','')
    # test_mmdet=Test_mmdet('3','','','')
    # test_mmdet=Test_mmdet('0','','','')
    # test_mmdet=Test_mmdet('17','','','')
    # test_mmdet=Test_mmdet('4','','','')
    # test_mmdet=Test_mmdet('20','','','')
    # test_mmdet=Test_mmdet('18',r'D:\work\data\microsoft\d_yt',
    #                       ['daowen', 'mianhua', 'penshabujun', 'aotuhen', 'aokeng', 'shuiyin', 'huanxingdaowen', 'pengshang', 'cashang', 'shahenyin', 'yiwu', 'tabian'],
    #                       r'D:\work\data\microsoft\d_yt\jsons\slect')
