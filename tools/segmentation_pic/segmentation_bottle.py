import json
import random
from numba import jit
from PIL import Image
import cv2 as cv
import time
import shutil


# @jit()
def seg():
    f = open('/home/zhang/datasets/bottle_seg/annotations/train1.json')
    js_labels = json.load(f)
    f.close()
    f__ = open('/home/zhang/datasets/bottle_seg/bottle_seg.json', 'w+')

    for js_it in js_labels['images']:
        num = 0
        if js_it['height'] < 1000:
            print(js_it['file_name'])
            shutil.copyfile('/home/zhang/datasets/bottle_seg/voc/images/' + js_it['file_name'],
                            '/home/zhang/datasets/bottle_seg/images/' + js_it['file_name'])
            # img = cv.imread('/home/zhang/datasets/bottle_seg/voc/images/' + js_it['file_name'])
            # cv.imwrite('/home/zhang/datasets/bottle_seg/images/'+ js_it['file_name'], img)
            continue
        # js_it['file_name'] = js_it['file_name'].replace('.jpg', str(num*250)+'.jpg')
        id_image = js_it['id']
        for annotations_image in js_labels['annotations']:
            if id_image == annotations_image['image_id']:
                w_ = random.randrange(50, 600)  # 658
                h_ = random.randrange(50, 442)  # 492
                w = annotations_image['bbox'][2]
                h = annotations_image['bbox'][3]
                if h > 390:
                    print('error1', end=' ')
                    print(annotations_image['category_id'], end=' ')
                    # print(js_it['name'], end=' ')
                    print(h)
                if w > 500:
                    print('error2', end=' ')
                    print(annotations_image['category_id'], end=' ')
                    print(w)
                y_min = int(annotations_image['bbox'][1] - h_)
                y_max = int(492 + y_min)
                x_min = int(annotations_image['bbox'][0] - w_)
                x_max = int(658 + x_min)
                if y_min <= 0:
                    # y_min = 0
                    # print("y_min = 0")
                    continue
                if x_min <= 0:
                    # x_min = 0
                    # print("x_min = 0")
                    continue
                if y_max >= int(js_it['height']):
                    # y_max = int(js_it['image_height'])
                    # print("y_max = int(js_it['image_height'])")
                    continue
                if x_max >= int(js_it['width']):
                    # x_max = int(js_it['image_width'])
                    # print("x_max = int(js_it['image_width'])")
                    continue
                img = cv.imread('/home/zhang/datasets/bottle_seg/voc/images/' + js_it['file_name'])
                cropped = img[y_min:y_max, x_min:x_max]
                newname = js_it['file_name'].replace('_', '_' +str(num)+'_')
                cv.imwrite('/home/zhang/datasets/bottle_seg/images/' + newname, cropped)

    f__.close()



        # start = time.time()
        # img = cv.imread('/home/zhang/datasets/cizhuan/images/' + js_it['name'])
        #
        # js_it['name'] = js_it['name'].replace('CAM', str(num))
        #
        # print(js_it['name'].replace('CAM', str(num)), end=" ")
        # js_it['image_width'] = 704
        # js_it['image_height'] = 544
        # js_it['bbox'][0] = w_
        # js_it['bbox'][1] = h_
        # js_it['bbox'][2] = w_ + w
        # js_it['bbox'][3] = h_ + h
        # json.dump(js_it, f__, ensure_ascii=False)
        # # num += 1
        # print(time.time() - start)



seg()
