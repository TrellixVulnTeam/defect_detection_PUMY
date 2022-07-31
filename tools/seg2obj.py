# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
i_ = 1
for class_ in range(10):
    class_+=1

    for label_ in ["Train", "Test"]:
        CLASS = "Class"
        CLASS = CLASS+str(class_)
        color_path = "/home/zhang/defect_detection/datasets/pic1/" + CLASS + "/" + CLASS + "/" + label_ + "/Label/"
        path_image = "/home/zhang/defect_detection/datasets/pic1/" + CLASS + "/" + CLASS + "/" + label_ + "/"
        path = "/home/zhang/defect_detection/datasets/pic1/" + CLASS + "/" + CLASS + "/" + label_ + "/Label/"
        label_txt = "/home/zhang/defect_detection/datasets/pic/labels/"
        image_label = "/home/zhang/defect_detection/datasets/pic/images_labels/"
        new_image = "/home/zhang/defect_detection/datasets/pic/images/"


        def find(img):
            img_flag = 255 * np.ones(img.shape, np.int8)
            count = 0
            findpoint = []
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if (img[x][y] == 255 and img_flag[x][y] == 255):
                        count += 1
                        img_flag[x][y] = count
                        findpoint.append((x, y))
                    while len(findpoint) > 0:
                        xx, yy = findpoint.pop()
                        if xx > 0:  # 上
                            if img[xx - 1][yy] == 255 and img_flag[xx - 1][yy] == 255:
                                findpoint.append((xx - 1, yy))
                                img_flag[xx - 1][yy] = count
                        if xx < img.shape[0] - 1:  # 下
                            if img[xx + 1][yy] == 255 and img_flag[xx + 1][yy] == 255:
                                findpoint.append((xx + 1, yy))
                                img_flag[xx + 1][yy] = count
                        if yy > 0:  # 左
                            if img[xx][yy - 1] == 255 and img_flag[xx][yy - 1] == 255:
                                findpoint.append((xx, yy - 1))
                                img_flag[xx][yy - 1] = count
                        if yy < img.shape[1] - 1:  # 右

                            if img[xx][yy + 1] == 255 and img_flag[xx][yy + 1] == 255:
                                findpoint.append((xx, yy + 1))
                                img_flag[xx][yy + 1] = count
            coutours = []
            for num in range(1, count + 1):
                coutours.append([])
                for x in range(img_flag.shape[0]):
                    for y in range(img_flag.shape[1]):
                        if img_flag[x][y] == num:
                            coutours[num - 1].append([x, y, img_flag[x][y]])
            desCoutous = {}
            ii = 0
            for num in range(len(coutours)):
                tmp = np.mat(coutours[num])
                minX = np.min(tmp[:, 0])
                maxX = np.max(tmp[:, 0])
                minY = np.min(tmp[:, 1])
                maxY = np.max(tmp[:, 1])
                dd = np.zeros((2, 2))
                dd[0][0] = minX
                dd[0][1] = maxX
                dd[1][0] = minY
                dd[1][1] = maxY
                if maxX - minX > 0 and maxY - minY > 0 and (maxX - minX) * (maxY - minY) >= 60:
                    desCoutous.update({ii: dd})
                    ii = ii + 1
            return desCoutous


        for f in os.listdir(path):
            if ".PNG" in f:
                img = cv2.imread(path + f, 0)
                img2_name = f.replace("_label", "")
                img2 = cv2.imread(path_image + img2_name)
                print(img2_name)
                # cv2.imwrite(new_image + str(i_) + ".PNG", img2)
                print(new_image + str(i_) + ".PNG")
                size = img.shape
                high = size[0]
                wight = size[1]
                find_results = find(img)
                if find_results:
                    img_color = cv2.imread(color_path + f)
                    # cv2.imwrite(image_label + str(i_) + ".PNG", img_color)
                    label_txt_all = label_txt + str(i_) + '.txt'
                    for i in range(len(find_results)):
                        str1 = str(int(class_ - 1)) + ' ' + str(int(find_results[i][1][0])/wight) + ' ' + str(
                            int(find_results[i][0][0])/high) + ' ' + str(int(find_results[i][1][1])/wight) + ' ' + str(
                            int(find_results[i][0][1])/high) + '\n'
                        with open(label_txt_all, 'a+') as txt:
                            txt.write(str1)
                        cv2.rectangle(img_color, (int(find_results[i][1][0]), int(find_results[i][0][0])),
                                      (int(find_results[i][1][1]), int(find_results[i][0][1])), (0, 255, 0), 1)

                # cv2.imwrite(image_label + str(i_) +".PNG", img_color)
                print(label_txt_all)
                print(new_image + str(i_) +".PNG")
                i_ = i_ + 1