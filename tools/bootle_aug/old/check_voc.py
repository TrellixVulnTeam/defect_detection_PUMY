import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import matplotlib.pyplot as plt
from math import sqrt as sqrt
print(os.getcwd())
# 需要检查的数据
sets = ['train', 'val', 'test']

# 需要检查的类别
classes = ['damage', 'deformation', 'broken_edge', 'spinning',
           'breakpoint', 'label_skew', 'label_wrinkling', 'label_bubble',
           'code_normal', 'code_exception']

if __name__ == '__main__':
    # GT框宽高统计
    width = []
    height = []

    for image_set in sets:
        # 图片ID不带后缀
        image_ids = open('ImageSets/%s.txt' % image_set).read().strip().split()
        for image_id in image_ids:
            # 图片的路径
            img_path = 'images/%s.jpg' % image_id
            # 这张图片的XML标注路径
            label_file = open('annotations/%s.xml' % image_id)
            tree = ET.parse(label_file)
            root = tree.getroot()
            try:
                size = root.find('size')    # 图像的size
                img_w = int(size.find('width').text)  # 宽
                img_h = int(size.find('height').text)  # 高
                img = cv2.imread(img_path)
            except:
                print(image_id)
                continue
            for obj in root.iter('object'):     # 解析object字段
                difficult = obj.find('difficult').text
                cls = obj.find('name').text #
                if cls not in classes or int(difficult) == 2:
                    continue
                cls_id = classes.index(cls)

                xmlbox = obj.find('bndbox')
                xmin = int(xmlbox.find('xmin').text)
                ymin = int(xmlbox.find('ymin').text)
                xmax = int(xmlbox.find('xmax').text)
                ymax = int(xmlbox.find('ymax').text)
                obj_w = xmax - xmin
                obj_h = ymax - ymin
                # width.append(w)
                # height.append(h)
                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)    # 对应目标上画框
                # resize图和目标框到固定值
                try:
                    w_change = (obj_w / img_w) * 416
                except:
                    print(image_id)
                h_change = (obj_h / img_h) * 416
                # width.append(w_change)
                # height.append(h_change)
                s = w_change * h_change
                width.append(sqrt(s))
                height.append(w_change / h_change)
            # print(img_path)
            img = cv2.resize(img, (608, 608))
            cv2.imshow(image_id, img)
            cv2.waitKey()
    plt.plot(width, height, 'ro')
    plt.show()
