import os
import json



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


def name():
    path = '/home/zhang/faiss_dataset'  # 图片原地址
    path1 = '/home/zhang/faiss_dataset'  # 图片现原地址
    i = 1
    for root, dirs, filelist in os.walk(path):  # root：当前读取到的文件夹 dirs：root中的子文件夹 filelist：root中的文件
        #filelist = os.listdir(self.path)  # 输出路径下所有文件的文件名
        for item in filelist:
            if item.endswith('.json'):
                json_file_path = os.path.join(os.path.abspath(root), item)  # os.path.abspath(self.path)读取path绝对路径
                instance = json_to_instance(json_file_path)
                instance['imagePath'] = '100_' + str(i) + '.jpg'
                instance_to_json(instance, json_file_path)
                dst = os.path.join(os.path.abspath(path1), '100_' + str(i) + '.json')  # os.path.join 路径拼接
                (filename, extension) = os.path.splitext(item)  # 分离找到的文件的名和后缀
                for item1 in filelist:
                    if item1.startswith(filename) & item1.endswith('.jpg'):
                        src1 = os.path.join(os.path.abspath(root), item1)  # os.path.abspath(self.path)读取path绝对路径
                        dst1 = os.path.join(os.path.abspath(path1), '100_' + str(i) + '.jpg')
                        try:
                            os.renames(src1, dst1)  # 用于file文件的重命名
                        except:
                            continue
                os.renames(json_file_path, dst)  # 用于file文件的重命名
                i = i + 1


name()
