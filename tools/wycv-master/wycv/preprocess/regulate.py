import os
import cv2
import json
import numpy as np


class Regulate:
    def __init__(self, img_list, json_list, start_point, crop_w, crop_h):
        self.img_list = img_list
        self.json_list = json_list
        self.start_point = start_point
        self.crop_w = crop_w
        self.crop_h = crop_h

    def process(self):
        h, w = self.img_list.shape[:2]
        img_filp = self.img_list
        json_filp = self.json_list
        if h < w:
            img_filp = self.img_flip()
            json_filp = self.json_flip()
        return self.cut_json(img_filp, json_filp, self.start_point[0], self.start_point[1], self.crop_w, self.crop_h)

    def json_flip(self):
        try:
            json_list = self.json_list
            json_list['imageHeight'], json_list['imageWidth'] = json_list['imageWidth'], json_list['imageHeight']
            for i in json_list['shapes']:
                new_q_points = []
                for j in i['points']:
                    x, y = j
                    new_q_points.append([y, x])
                i['points'] = new_q_points
            return json_list
        except Exception as e:
            print('json flip error')

    # 横转竖立
    def img_flip(self):
        try:
            img_list = self.img_list
            # 获取输入图像的信息，生成旋转操作所需的参数（padding: 指定零填充的宽度； canter: 指定旋转的轴心坐标）
            h, w = img_list.shape[:2]
            padding = (w - h) // 2
            center = (w // 2, w // 2)
            # 在原图像两边做对称的零填充，使得图片由矩形变为方形
            img_padded = np.zeros(shape=(w, w, 3), dtype=np.uint8)
            img_padded[padding:padding + h, :, :] = img_list
            # 逆时针-90°(即顺时针90°)旋转填充后的方形图片
            M = cv2.getRotationMatrix2D(center, 90, 1)
            rotated_padded = cv2.warpAffine(img_padded, M, (w, w))
            # 从旋转后的图片中截取出我们需要的部分，作为最终的输出图像
            output = rotated_padded[:, padding:padding + h, :]
            output = cv2.flip(output, 0)
            return output
        except Exception as e:
            print('img flip error')

    def get_new_img(self, img_data, x_min, y_min, x_max, y_max):
        # 切图并保存
        # print('x_min: ', x_min, 'y_min: ', y_min, 'x_max:', x_max, 'y_max:', y_max)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        left, top, right, down = 0, 0, 0, 0  # need padding size
        img_crop = img_data[y_min:y_max, x_min:x_max]
        # ret = cv2.copyMakeBorder(img_crop, top, down, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))#padding
        return img_crop

    def cut_json(self, img_data, json_data, x, y, w, h):
        json_data['imageHeight'] = h
        json_data['imageWidth'] = w
        index_remove = []
        for index, shape in enumerate(json_data['shapes']):
            points = shape['points']
            new_points = []
            for i in points:
                n_p = [i[0] - x, i[1] - y]
                if 0<=n_p[0]<=765 and 0<=n_p[1]<=765:
                    new_points.append(n_p)
            if len(new_points) == 0:
                index_remove.append(index)
            else:
                shape['points'] = new_points
        index_remove.reverse()
        for index in index_remove:
            json_data['shapes'].pop(index)
        img_data = self.get_new_img(img_data, x, y, x + w, y + h)
        return img_data, json_data
