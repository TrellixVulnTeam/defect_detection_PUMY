import numpy as np
import cv2
import random
import os
import time 
import datetime
import os.path as osp
import shutil
import json
from utils import *
from xml.etree import ElementTree as ET
from PIL import Image
# Fs_Root_path="/home/adt/data/data/Cjian/demo机/remove_tiny20210305/damian/hengtu/select_other/train/pr2/"
# sourceDirPath=Fs_Root_path+"/crop/"
# outimgDirPath=Fs_Root_path+"/select_daowen/"
# outimgDirPath2=Fs_Root_path+"/select_other/"
# # outlabelDirPath=Fs_Root_path+"/label2/"
# cut_label=['daowen','aotuhen','pengshabujun','heidian']
# cut_label=[0]
def select_defect_image_json(sourceDirPath,outimgDirPath,outimgDirPath2,cut_label,replace_label,target_label):
    if not osp.exists(outimgDirPath):
        print("outimgDirPath",outimgDirPath)
        os.mkdir(outimgDirPath)
    if not osp.exists(outimgDirPath2):
        print("outimgDirPath2",outimgDirPath2)
        os.mkdir(outimgDirPath2)

    json_name=[]
    for files in os.listdir(sourceDirPath):
        if not files.endswith('.json'): continue
        json_name.append(files)
    # sourceFiles = sorted(os.listdir(sourceDirPath))

    for sourceName in json_name:
        print("sourceName::",sourceName)
        json_p=os.path.join(sourceDirPath,sourceName)
        with open(json_p, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        img_n = os.path.join(sourceDirPath,json_data['imagePath'])#原图像名
        if not osp.exists(img_n):continue
        flag=0
        for i in json_data['shapes']:
            if i['label'] in cut_label:
                flag = 1
            # xywh= points_to_xywh(i)
            # if len(xywh)==0:
            #     continue
            # if (xywh[2]>500 or xywh[3]>500) and i['label']=='yise' :
            #     flag = 1


            # if (xywh[2]>50 or xywh[3]>50) and (i['label'].startswith('loushi') or  i['label'].startswith('hard')):
            #     flag = 1
        if flag==1:
            print("flag::",flag)
            out_json_Path = os.path.join(outimgDirPath, sourceName)

            out_image_Path = os.path.join(outimgDirPath, json_data['imagePath'])
            out_xml_path=out_json_Path.split('.')[0]+'.jpg'
            xml_name=json_p.split('.')[0]+'.jpg'
            # shutil.copyfile(xml_name, out_xml_path)
            shutil.copyfile(img_n, out_image_Path)
            shutil.copyfile(json_p, out_json_Path)
        # else:
        #     print("flag::",flag)
        #     out_json_Path = os.path.join(outimgDirPath2, sourceName)
        #     out_image_Path = os.path.join(outimgDirPath2, json_data['imagePath'])
        #     out_xml_path=out_json_Path.split('.')[0]+'.xml'
        #     xml_name=json_p.split('.')[0]+'.xml'
        #     shutil.copyfile(img_n, out_image_Path)
        #     shutil.copyfile(json_p, out_json_Path)
        #     shutil.copyfile(xml_name, out_xml_path)
def get_box(txt_file_path,cut_label,replace_label,target_label):

    boxes = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return boxes
    # 遍历txt文件中的每一个检测目标
    for line in lines:
        words = line.split(' ')
        print("word:",words[0])
        if words[0] in cut_label:
            cx, cy, w, h = float(words[1]), float(words[2]), float(words[3]), float(words[4])
            # 一个Box代表一个检测目标的xywh、label、confidence
            if words[0] in replace_label:
                label=target_label
            else:
                label=words[0]
            boxes.append(Box(category=(int(label)), x=cx, y=cy, w=w, h=h))
    return boxes
def save_txt_img(box,outtxtDirPath,sourceimgDirPath,outimgDirPath):
    if not os.path.exists(sourceimgDirPath):
        box_info = " "
        return box_info
    if len(box)==0:
        box_info = " "
        shutil.copyfile(sourceimgDirPath, outimgDirPath)
        return box_info
    else:
        with open(outtxtDirPath, 'w') as f:
            for i in range(len(box)):


                    # box_info = "%d %.03f %.03f %.03f %.03f" % (label_dic.index(obj['label']), cx/width, cy/height, W_width, H_height)
                box_info = "%d %.03f %.03f %.03f %.03f" % (int(box[i].category), float(box[i].x), float(box[i].y), float(box[i].w), float(box[i].h))
                f.write(box_info)
                f.write('\n')
        shutil.copyfile(sourceimgDirPath, outimgDirPath)

def select_defect_image_txt(sourceDirPath,outDirPath,outimgDirPath2,cut_label={'0','1','2','3'},replace_label={'0','1','2','3'},target_label=0):
    if not osp.exists(outDirPath):
        print("outDirPath",outDirPath)
        os.mkdir(outDirPath)
    if not osp.exists(outimgDirPath2):
        print("outimgDirPath2",outimgDirPath2)
        os.mkdir(outimgDirPath2)

    json_name=[]
    for files in os.listdir(sourceDirPath):
        if not files.endswith('.txt'): continue
        json_name.append(files)

    for sourceName in json_name:
        txt_file_path = os.path.join(sourceDirPath, sourceName)

        # img = Image.open(txt_file_path)
        box=get_box(txt_file_path,cut_label,replace_label,target_label)
        outtxtDirPath=os.path.join(outDirPath,sourceName)
        outimgDirPath=outtxtDirPath.split('.')[0]+'.jpg'
        sourceimgDirPath=txt_file_path.split('.')[0]+'.jpg'

        save_txt_img(box,outtxtDirPath,sourceimgDirPath,outimgDirPath)

def select_defect_image_xml(xml_folder_path,sourceDirPath,outimgDirPath,outimgDirPath2,cut_label):
    if not osp.exists(outimgDirPath):
        print("outimgDirPath",outimgDirPath)
        os.mkdir(outimgDirPath)
    if not osp.exists(outimgDirPath2):
        print("outimgDirPath2",outimgDirPath2)
        os.mkdir(outimgDirPath2)


    img_files = os.listdir(sourceDirPath)
    # 遍历img
    for img_file in img_files:
        img_file_path = os.path.join(sourceDirPath, img_file)
        # 过滤文件夹和非图片文件
        if not os.path.isfile(img_file_path) or img_file[img_file.rindex('.')+1:] not in IMG_TYPES: continue
        xml_file_path = os.path.join(xml_folder_path, img_file[:img_file.rindex('.')] + '.xml')
        try:
            root = ET.parse(xml_file_path).getroot()
        except Exception as e:
            # 若为无目标图片
            print('\033[1;33m%s has no xml file in %s. So saved as an empty json.\033[0m' % (img_file, xml_folder_path))
            continue
        item_name='item'
        items = root.iter(item_name)
        flag=0
        for item in items:
            obj = {'label': word_to_pinyin(item[0].text)}
            label = word_to_pinyin(item[0].text)
            if label in cut_label:
                flag=1
            if item.find('bndbox') != None:
                xys = extract_xys(item.find('bndbox'))
                obj['shape_type'] = 'rectangle'
                obj['points'] = [[xys[0], xys[1]], [xys[2], xys[3]]]
            elif item.find('point') != None:
                xys = extract_xys(item.find('point'))
                obj['shape_type'] = 'point'
                obj['points'] = [[xys[0], xys[1]]]
            elif item.find('polygon') != None:
                xys = extract_xys(item.find('polygon'))
                obj['shape_type'] = 'polygon'
                obj['points'] = [[xys[i-1], y] for i,y in enumerate(xys) if i%2==1]
            elif item.find('line') != None:
                xys = extract_xys(item.find('line'))
                # 排除标注小组的重复落点
                points = [[xys[i-1], y] for i,y in enumerate(xys) if i%2==1]
                points_checked = [points[0]]
                for point in points:
                    if point != points_checked[-1]:
                        points_checked.append(point)
                obj['points'] = points_checked
                if len(points_checked) == 1:
                    obj['shape_type'] = 'point'
                elif len(points_checked) == 2:
                    obj['shape_type'] = 'line'
                else:
                    obj['shape_type'] = 'linestrip'
            else:
                print('Please check the xml file to add polygon type!')
                exit(0)
            xywh= points_to_xywh(obj)
            if len(xywh)==0:
                continue
            # if (xywh[2]>50 or xywh[3]>50):
            #     flag = 1
            if (xywh[2]>200 or xywh[3]>200) and (obj['label']=='yise'):
                flag = 1
        if flag==1:
            out_image_Path = os.path.join(outimgDirPath, img_file)

            out_xml_path = os.path.join(outimgDirPath, img_file.split('.')[0]+'.xml')
            # out_xml_path = out_json_Path.split('.')[0] + '.jpg'
            # xml_name = json_p.split('.')[0] + '.jpg'
            # shutil.copyfile(xml_name, out_xml_path)
            shutil.copyfile(img_file_path, out_image_Path)
            shutil.copyfile(xml_file_path, out_xml_path)
            print("out_xml_path:",out_xml_path)


Fs_Root_path="/home/adt/data/data/weiruan/weiruan_a/"
sourceDirPath=Fs_Root_path+"/select_daowen_new/"
xml_folder_path=sourceDirPath+'/select_daowen_new/'
outimgDirPath=Fs_Root_path+"/select_daowen_train/"
outimgDirPath2=Fs_Root_path+"/select_other_d/"
# outlabelDirPath=Fs_Root_path+"/label2/"
# cut_label=['pengshang','tabian']

# cut_label=['loushi_daowen','loushi_pengshang','loushi_tabian','guojian_daowen','guojian_pengshang','guojian_tabian']
cut_label={'daowen'}
# cut_label={'0','1','2','3','4','5'}
replace_label={'2'}
target_label='1'
select_defect_image_json(sourceDirPath,outimgDirPath,outimgDirPath2,cut_label,replace_label,target_label)
