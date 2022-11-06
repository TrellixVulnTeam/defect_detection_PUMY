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


# 检测json里面的imageData有没有值
# txt_file=open('results.txt',mode='w+')
# results_path = '/home/ubuntu/Code/tools/new/'
# os.makedirs(results_path)
path = '/home/zhang/datasets/floor_cut_blance/source/train/'
classes = {}
for root, sub_folder, files in os.walk(path):
    for json_file in files:
        if not json_file.endswith('.json'): continue
        # print(json_file)
        #
        if json_file == 'Defect_properties.json' or json_file == 'Model_file.json' or json_file == 'Optics_file.json':continue
        json_file_path = os.path.join(root, json_file)
        instance = json_to_instance(json_file_path)
        # print(len(instance['shapes']))
        # print(instance['shapes'][0]['label'])
        for i in range(len(instance['shapes'])):
            # print(i)
            if not instance['shapes'][i]['label'] in classes:
                classes[instance['shapes'][i]['label']] = 1
                if instance['shapes'][i]['label'] == 'slot ash':
                    print(json_file_path)
                # classes.append(instance['shapes'][0]['label'])
            else:
                classes[instance['shapes'][i]['label']] += 1
print(classes)
print(len(classes))
