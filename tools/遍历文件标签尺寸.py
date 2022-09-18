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


# class_list = ['white-impurities', 'black-spot', 'edge-damage', 'bubble-gum']
class_list = ['edge-damage']


def labelSize(path):
    files = os.listdir(path)
    count = 0
    x_max = []
    y_max = []
    for file in files:
        if file.endswith('.json'):
            instance = json_to_instance(os.path.join(path, file))
            for shape in instance['shapes']:
                if shape['label'] in class_list:
                    point_x = []
                    point_y = []
                    for point in shape['points']:
                        point_x.append(point[0])
                        point_y.append(point[1])
                    labels_width = max(point_x) - min(point_x)
                    labels_high = max(point_y) - min(point_y)
                    x_max.append(labels_width)
                    y_max.append(labels_high)
                    if labels_high > 160 or labels_width > 160:
                        print('labels width = ', labels_width, '  ', 'labels high = ', labels_high, ' ', count,shape['label'])
                        # print(instance['shapes'][0]['label'])
                        count += 1

                # x_max = sorted(x_max, reverse=True)
                #
                # y_max = sorted(y_max, reverse=True)
    # print(max(x_max), max(y_max))


if __name__ == '__main__':
    work_dir = '/home/zhang/datasets/floor_cut/source/train'
    labelSize(work_dir)
