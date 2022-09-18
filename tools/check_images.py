import os
import cv2
import numpy as np

path = "/home/zhang/Project/Huawei_c/DM/"
imglist = os.listdir(path)
count = 1
for fi in imglist:
    if fi.endswith('.jpg'):
        img_folder = os.path.join(path + fi)
        # print(img_folder)
        image = cv2.imdecode(np.fromfile(img_folder, dtype=np.uint8), -1)
        try:
            image.shape
            print(img_folder, image.shape, count)
            count += 1
        except:
            print('fail to read', img_folder, count)
            # count += 1
            os.remove(img_folder)
            continue
