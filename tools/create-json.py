import os
import json
import cv2
import numpy as np

def instance_to_json(instance, json_file_path):
    '''
    :param instance: json instance
    :param json_file_path: 保存为json的文件路径
    :return: 将json instance保存到相应文件路径
    '''
    with open(json_file_path, 'w', encoding='utf-8') as f:
        content = json.dumps(instance, ensure_ascii=False, indent=2)
        f.write(content)

def create_empty_json_instance(img_file_path: str,dict1):
    '''
    :param img_file_path: img路径
    :return: 构建一个空的labelme json instance对象
    '''
    print('**********',img_file_path)
    instance = {'version': '1.0',
                'imageData':None,
                'imageWidth':'',
                'imageHeight':'',
                'imageDepth':'',
                'imagePath': img_file_path[img_file_path.rindex(os.sep) + 1:],
                'shapes': [dict1],
                }
    img = cv2.imread(img_file_path)
    instance['imageHeight'], instance['imageWidth'], instance['imageDepth'] = img.shape
    instance_to_json(instance, img_file_path[:img_file_path.rindex('.')]+'.json')
    #print(img_file_path[:img_file_path.rindex('.')]+'.json')
    return instance


for _,_,files in os.walk('result/'):
    for img_file in files:
        img_file_path = os.path.join('result/', img_file)
        img = cv2.imread(img_file_path,0)
        gray=cv2.GaussianBlur(img,(3,3),1.5)
        ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        contours, hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(0,0,255),1)
        dict1 = {'label': 'liewen', 'shape_type': 'polygon', 'points': [], 'status': '0', 'describe': 'biaozhurending','mlevel':None,'plevel':None}

        max1=80
        dict=[]
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area>max1:
                max1=area
                a=contours[i]
                b=a.reshape(contours[i].shape[0],contours[i].shape[2])
                dict1['points'] = b.tolist()
        create_empty_json_instance(img_file_path,dict1)




        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
