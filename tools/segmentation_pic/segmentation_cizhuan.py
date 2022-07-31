import json
import random
from numba import jit
from PIL import Image
import cv2 as cv
import time

# @jit()
def seg():
    f = open('/home/zhang/defect_detection/tools/segmentation_pic/train_annos.json')
    js_labels = json.load(f)
    f.close()
    f__ = open('train_annos_new.json', 'w+')
    num = 1
    for js_it in js_labels:
        start = time.time()
        img = cv.imread('/home/zhang/datasets/cizhuan/images/' + js_it['name'])
        w_ = random.randrange(50, 650)  # 704
        h_ = random.randrange(50, 500)  # 544
        w = js_it['bbox'][2] - js_it['bbox'][0]
        h = js_it['bbox'][3] - js_it['bbox'][1]
        if h > 500:
            print('error6', end=' ')
            print(js_it['name'], end=' ')
            print(h)
        if w > 600:
            print('error7', end=' ')
            print(js_it['name'], end=' ')
            print(w)
        y_min = int(js_it['bbox'][1] - h_)
        y_max = int(544 + y_min)
        x_min = int(js_it['bbox'][0] - w_)
        x_max = int(704 + x_min)
        if y_min <= 0:
            # y_min = 0
            # print("y_min = 0")
            print('error1')
            continue
        if x_min <= 0:
            # x_min = 0
            # print("x_min = 0")
            print('error2')
            continue
        if y_max >= int(js_it['image_height']):
            # y_max = int(js_it['image_height'])
            # print("y_max = int(js_it['image_height'])")
            print('error3')
            continue
        if x_max >= int(js_it['image_width']):
            # x_max = int(js_it['image_width'])
            # print("x_max = int(js_it['image_width'])")
            print('error4')
            continue
        js_it['image_width'] = 704
        js_it['image_height'] = 544
        js_it['bbox'][0] = w_
        js_it['bbox'][1] = h_
        js_it['bbox'][2] = w_ + w
        js_it['bbox'][3] = h_ + h
        if js_it['bbox'][2] > 704 or js_it['bbox'][3] > 544:
            print('error5')
            continue
        cropped = img[y_min:y_max,
                  x_min:x_max]
        js_it['name'] = js_it['name'].replace('CAM', str(num))
        cv.imwrite('/home/zhang/defect_detection/tools/segmentation_pic/images/' + js_it['name'], cropped)
        print(js_it['name'].replace('CAM', str(num)),end=" ")

        json.dump(js_it, f__, ensure_ascii=False)
        num += 1
        print(time.time()-start)
    f__.close()

seg()
