import os
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
import os
import json
import shutil
import glob

classes = {}
work_dir = '/home/zhang/Project/Apple/DM'


del_ = True  # delete


def json_to_instance(json_file_path):
    '''
    :param json_file_path: json文件路径
    :return: json instance
    '''
    with open(json_file_path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    return instance


def instance_to_json(instance, json_file_path):
    '''
    :param instance: json instance
    :param json_file_path: 保存为json的文件路径
    :return: 将json instance保存到相应文件路径
    '''
    with open(json_file_path, 'w', encoding='utf-8') as f:
        content = json.dumps(instance, ensure_ascii=False, indent=2)
        f.write(content)


def checkImgAndLabel(img, json):
    classes = {}
    image = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
    instance = json_to_instance(json)
    try:
        image.shape
        instance['shapes'][0]
        if 'imageData' in instance:
            if type(instance['imageData']) == str:
                instance['imageData'] = None
                instance_to_json(instance, json)
        for i in range(len(instance['shapes'])):
            if not instance['shapes'][i]['label'] in classes:
                classes[instance['shapes'][i]['label']] = 1
                # classes.append(instance['shapes'][0]['label'])
            else:
                classes[instance['shapes'][i]['label']] += 1
    except:
        print('error', image, json)
        if del_:
            os.remove(img)
            os.remove(json)

    return classes


def checkData(path):
    classes_ = {}
    inter_list = {}
    process_num = 10
    process_pool = multiprocessing.Pool(processes=process_num)
    cv2.setNumThreads(0)
    img_list = []
    json_list = []
    xml_list = []
    for root, sub_folder, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                img_list.append(os.path.join(root, file))
            elif file.endswith('.json'):
                json_list.append(os.path.join(root, file))
            elif file.endswith('.xml'):
                xml_list.append(os.path.join(root, file))
    for img_iter in range(len(img_list) - 1, -1, -1):
        if not img_list[img_iter].replace('.jpg', '.json') in json_list:
            print(img_list[img_iter], 'no label')
            if del_:
                os.remove(img_list[img_iter])
            img_list.remove(img_list[img_iter])


    for label_iter in range(len(json_list) - 1, -1, -1):
        if not json_list[label_iter].replace('.json', '.jpg') in img_list:
            print(json_list[label_iter], 'no label')
            if del_:
                os.remove(json_list[label_iter])
            json_list.remove(json_list[label_iter])

    # print(json_list)
    if not len(json_list) == len(img_list):
        print('error error error error error error error error error error error error error error error error error '
              'error error error error ')
        return
    json_list.sort()
    img_list.sort()
    pbar = tqdm(total=len(json_list))
    for i in range(len(json_list)):
        classes__ = process_pool.apply_async(checkImgAndLabel, args=(img_list[i], json_list[i],),
                                             callback=lambda _: pbar.update()).get()
        for iter_ in classes__:
            # print(classes__, iter_, classes__[iter_])
            if not iter_ in classes_:
                classes_[iter_] = classes__[iter_]
            else:
                classes_[iter_] += classes__[iter_]
    process_pool.close()
    process_pool.join()
    # print(len(img_list),len(json_list))
    # pbar = tqdm(total=len(source_json_list))
    return classes_

if __name__ == '__main__':
    result = checkData(work_dir)
    print(result)
    print(len(result))
