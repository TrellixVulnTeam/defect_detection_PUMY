import os
import json
import shutil
import glob


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


count = 1
# 检测json里面的imageData有没有值
# txt_file=open('results.txt',mode='w+')
# results_path = '/home/ubuntu/Code/tools/new/'
# os.makedirs(results_path)
path = '/home/zhang/Project/Huawei_AAC/GJ'
del_flag = False
for root, sub_folder, files in os.walk(path):
    for json_file in files:
        if not json_file.endswith('.json'): continue
        if json_file == 'Defect_properties.json' or json_file == 'Model_file.json' or json_file == 'Optics_file.json': continue
        json_file_path = os.path.join(root, json_file)
        instance = json_to_instance(json_file_path)
        for i in range(len(instance['shapes'])):
            if instance['shapes'][i]['label'] == 'AYS':
                # del_flag = True
                print(json_file_path, count)
                count += 1
                instance['shapes'][i]['label'] = 'yise'
                instance_to_json(instance, json_file_path)
        if del_flag:
            os.remove(json_file_path)
            print('remove', json_file_path)
            del_flag = False
        # if 'imageData' in instance:
        #     if type(instance['imageData']) == str:
        #         print(json_file_path, 'is error',count)
        #         # print(type(instance['imageData']))
        #         # print(os.path.join(curDir, json_file))
        #         # path1 = os.path.join(curDir, json_file)
        #         # shutil.copy(path1, results_path)
        #         # path_jpg = path1[:-4] + 'jpg'
        #         # shutil.copy(path_jpg, results_path)
        #         instance['imageData'] = None
        #         instance_to_json(instance, json_file_path)

        # txt_file.write(path1+'\n')
# txt_file.close()
# 检测json里面的imageData有没有值
