import shutil
import os
import json
import cv2

import multiprocessing


def json_to_instance(json_file_path):
    '''
    :param json_file_path: json文件路径
    :return: json instance
    '''
    with open(json_file_path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    return instance


tar_name = ['640plus/', '768/']
path = '/home/zhang/datasets/floor_cut/images/'
tar = '/home/zhang/datasets/floor_cut/'

count = 1


def moveByLabelShape(img_, tar_, tar_name_):
    if img_.endswith('.json'):
        instance = json_to_instance(img_)
        for shape in instance['shapes']:
            point_x = []
            point_y = []
            for point in shape['points']:
                point_x.append(point[0])
                point_y.append(point[1])
            labels_width = max(point_x) - min(point_x)
            labels_high = max(point_y) - min(point_y)

            if labels_high >= 600 or labels_width >= 600:
                shutil.copy(img_, tar_ + tar_name_[0] + img_.split('/')[-1])
                shutil.copy(img_.replace('.json', '.bmp'), tar_ + tar_name_[0] + img_.split('/')[-1].replace('.json', '.bmp'))
                print(img_, tar_ + tar_name_[0] + img_.split('/')[-1])
                print(img_.replace('.json', '.bmp'), tar_ + tar_name_[0] + img_.split('/')[-1].replace('.json', '.bmp'))
                break


for root, sub_folder, files in os.walk(path):
    for file in files:
        moveByLabelShape(os.path.join(root, file), tar, tar_name)
        count += 1
