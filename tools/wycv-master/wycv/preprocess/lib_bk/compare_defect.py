#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) micro-i.com.cn, Inc. All Rights Reserved
#
"""
# This module provide
# Authors: Kerry(junyou.zheng@micro-i.com.cn)
# Date: 2021/1/22 上午11:49
"""
import json
import glob
import os
from shutil import move, copyfile

def instance_to_json(instance, json_file_path: str):
    '''
    :param instance: json instance
    :param json_file_path: 保存为json的文件路径
    :return: 将json instance保存到相应文件路径
    '''
    with open(json_file_path, 'w', encoding='utf-8') as f:
        content = json.dumps(instance, ensure_ascii=False, indent=2)
        f.write(content)

def json_to_instance(json_file_path: str):
    '''
    :param json_file_path: json文件路径
    :return: json instance
    '''
    with open(json_file_path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    return instance

def compare_defect(json_file_path, defect_list, to_path):
    jpg_file_path = json_file_path.split('.')[0]+'.jpg'

    instance = json_to_instance(json_file_path)
    for obj in instance['shapes']:
        # if obj["label"] in defect_list:
        # if obj["label"].endswith('pengshang'):
        # obj["group_id"] == 99:
        try:
            if obj["label"].startswith('loushi') or obj["label"].startswith('hard') :#and obj["label"].startswith('loushi')
                print(json_file_path)
                move(json_file_path, to_path)
                move(jpg_file_path, to_path)
                break
        except:
            pass
            #with open('a.txt', 'a') as f:
            #    f.write(json_file_path+ '\n')
defect_list         = ['guashang']

json_path           = '/home/adt/data/data/weiruan/weiruan_a/cemian_guaijiao/0408/select/pr/'
to_path             = '/home/adt/data/data/weiruan/weiruan_a/cemian_guaijiao/0408/select/out3/'

if not os.path.exists(to_path):
    os.makedirs(to_path)

json_list = glob.glob(json_path + '*.json')
for json_file_path in json_list:
    json_name = os.path.basename(json_file_path)
    compare_defect(json_file_path, defect_list, to_path)
