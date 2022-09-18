# ---------------------------class2----------------------start
# 切除图像多余边框
import json
import os
import cv2

'''
@Description: 
            主要功能：切除非光学面区域
            关键处理：有标注切除，无标注切除
@author: lijianqing
@date: 2020/11/24 16:45
@return 

@param: 输入：数据存放路径，root_path 
             切除后数据保存路径：out_path（不存在的时候会自动创建）
             切图的起点位置：start_point=[96,46] 
             切图宽高：crop_w = 3308,crop_h = 4854
             切图是否有标注：anno=True，没有标注时为False
'''


class Cut_edge(object):
    def __init__(self, root_path, out_path, start_point=[150, 0], crop_w=4500, crop_h=6000, anno=True):
        # root_path = r'C:\Users\xie5817026\Desktop\damian\jalama\jalama',labelme格式的数据路径，若有标注数据时同时存放img和json文件,若无标注信息时只处理图
        # out_path = r'C:\Users\xie5817026\Desktop\damian\jalama\jalama',切图后的路径，有则直接保存，无则创建路径后再保存。
        # #start_point=[1004,66]#shu
        # #start_point=[321,778]#hen
        # start_point=[96,46]#hen wx
        # #start_point=[96,46]#hen wx
        # print(type(start_point[0]),'--')
        # crop_w = 3308
        # crop_h = 4854
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in os.listdir(root_path):
            if i.endswith('14.json' or '14.jpg'):
                start_point = [150, 0]  # 14号图，切左右
            elif i.endswith('13.json' or '13.jpg'):
                start_point = [0, 24]  # 13号图，切上下
            if anno:
                if i.endswith('.json'):
                    jsons_path = os.path.join(root_path, i)
                    self.cut_json(jsons_path, root_path, out_path, start_point, crop_w, crop_h)
            else:
                if i.endswith('.jpg'):
                    img_n = os.path.join(root_path, i)  # 原图像名
                    print('img_n', img_n)
                    img_np = cv2.imread(img_n)  # 原图数据
                    self.save_new_img(img_np, i, start_point[0], start_point[1], start_point[0] + crop_w,
                                      start_point[1] + crop_h, out_path)

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)

    def save_new_img(self, img_np, img_name, xmin, ymin, xmax, ymax, out_path):
        # 切图并保存
        print('-111-', xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        left, top, right, down = 0, 0, 0, 0  # need padding size
        img_crop = img_np[ymin:ymax, xmin:xmax]
        # ret = cv2.copyMakeBorder(img_crop, top, down, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))#padding
        cv2.imwrite(os.path.join(out_path, img_name), img_crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return 0

    def get_new_location(self, point, start_point):
        print(start_point, '--')
        print(type(start_point[0]), '---', type(point[0]))
        return [point[0] - start_point[0], point[1] - start_point[1]]

    def cut_json(self, json_p, img_sourc, out_path, start_point, w, h):
        with open(json_p, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        img_n = os.path.join(img_sourc, json_data['imagePath'])  # 原图像名
        print('img_n', img_n)
        img_np = cv2.imread(img_n)  # 原图数据
        json_data['imageHeight'] = h
        json_data['imageWidth'] = w
        for index_object in json_data['shapes']:
            points = index_object['points']
            new_points = []
            for i in points:
                n_p = self.get_new_location(i, start_point)
                new_points.append(n_p)
            index_object['points'] = new_points
        self.save_new_img(img_np, json_data['imagePath'], start_point[0], start_point[1], start_point[0] + w,
                          start_point[1] + h, out_path)
        new_name_json = json_data['imagePath'].replace('jpg', 'json')   # 注意bug: 有的imagePath需要分割取文件名
        self.save_json(json_data, os.path.join(out_path, new_name_json))


# ---------------------------class2----------------------end

# ---------------------------class3----------------------start
# 将横图转竖图，或将横图及横图标注转竖图或竖图标注
# 输入：数据路径，保存路径，是否有标注
# HorizontalToVertical(root_path,save_path,anno=True)

import json
import multiprocessing
import time
import numpy as np
import cv2
import os


class HorizontalToVertical(object):
    def __init__(self, img_p, save_p, anno=True):
        # img_p = r'C:\Users\xie5817026\Desktop\damian\jalama\jalama'#数据路径，数据可以时标注和图像一起，也可以时只有图像数据
        # save_p = r'C:\Users\xie5817026\Desktop\damian\jalama'#保存路径
        # anno 有标注为True,无标注为False
        jsons_l = []
        imgs_l = []
        if not os.path.exists(save_p):
            os.makedirs(save_p)
        for i in os.listdir(img_p):
            if i.endswith('json'):
                s_p = os.path.join(save_p, i)
                i_p = os.path.join(img_p, i)
                jsons_l.append((i_p, s_p))
            elif i.endswith('.jpg'):
                s_i_p = os.path.join(save_p, i)
                i_i_p = os.path.join(img_p, i)
                imgs_l.append((i_i_p, s_i_p))
        pool = multiprocessing.Pool(processes=16)  # 创建进程个数
        pool1 = multiprocessing.Pool(processes=16)  # 创建进程个数
        if anno:
            start_time = time.time()
            pool.map(self.jsons_flip, jsons_l)
            print('run time:', time.time() - start_time)
            pool.close()
            pool.join()
        start_time1 = time.time()
        pool1.map(self.img_flip, imgs_l)
        pool1.close()
        pool1.join()
        print('run time:', time.time() - start_time1)

    def save_json(self, dic, save_path):
        json.dump(dic, open(save_path, 'w', encoding='utf-8'), indent=4)

    def parse_para(self, input_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        return ret_dic

    def jsons_flip(self, l):
        try:
            json_one, op = l
            json_1 = self.parse_para(json_one)
            json_1['imageHeight'], json_1['imageWidth'] = json_1['imageWidth'], json_1['imageHeight']
            for i in json_1['shapes']:
                new_q_points = []
                for j in i['points']:
                    x, y = j
                    new_q_points.append([y, x])
                i['points'] = new_q_points
            if not os.path.isfile(op):
                self.save_json(json_1, op)
        except:
            shutil.move(json_one, r"D:\data\module-c\remove-tiny\damian\train\except_jsons")
            print('异常处理：', json_one)

    # 横转竖立
    def img_flip(self, l):
        try:
            img_path, save_path = l
            print(img_path)
            # 读取原图像
            img = cv2.imread(img_path)
            print('img.shape', img.shape)
            # 获取输入图像的信息，生成旋转操作所需的参数（padding: 指定零填充的宽度； canter: 指定旋转的轴心坐标）
            h, w = img.shape[:2]
            padding = (w - h) // 2
            center = (w // 2, w // 2)
            # 在原图像两边做对称的零填充，使得图片由矩形变为方形
            img_padded = np.zeros(shape=(w, w, 3), dtype=np.uint8)
            img_padded[padding:padding + h, :, :] = img
            # 逆时针-90°(即顺时针90°)旋转填充后的方形图片
            M = cv2.getRotationMatrix2D(center, 90, 1)
            rotated_padded = cv2.warpAffine(img_padded, M, (w, w))
            # 从旋转后的图片中截取出我们需要的部分，作为最终的输出图像
            output = rotated_padded[:, padding:padding + h, :]
            output = cv2.flip(output, 0)
            if not os.path.isfile(save_path):
                cv2.imwrite(save_path, output, [cv2.IMWRITE_JPEG_QUALITY, 100])
        except:
            shutil.move(img_path, r"D:\data\module-c\remove-tiny\damian\train\except_jsons")
            print('异常处理：', img_path)


# ---------------------------class3----------------------end

