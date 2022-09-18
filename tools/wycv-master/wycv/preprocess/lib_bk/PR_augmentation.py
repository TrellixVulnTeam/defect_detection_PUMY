from utils import *
import shutil

label_dict={'baisezaodian', 'heidian', 'aokeng', 'yiwu', 'cashang'}
def precision_recall_visualize(target_folder_path, img_boxes_query, saved_folder_name, iou_thres=0.1, hard_thres=0.7,guo_thres=0.3, recall=True, precision=True):
    '''
    :param target_folder_path: json文件夹路径
    :param img_boxes_query: 该方法根据json文件名返回box列表
    :return: 可视化漏失、过检结果
    '''
    # 漏失、过检文件夹
    pr_folder_path = os.path.join(target_folder_path, saved_folder_name)
    if not os.path.exists(pr_folder_path): os.makedirs(pr_folder_path)
    img_files = os.listdir(target_folder_path)
    labels_tatal = {}
    lou_tatal = {}
    guo_tatal={}
    total_objs = 0
    over_detect = 0
    no_detect = 0
    # 遍历img
    for img_file in img_files:
        img_file_path = os.path.join(target_folder_path, img_file)
        # 过滤文件夹和非图片文件
        if not os.path.isfile(img_file_path) or img_file[img_file.rindex('.')+1:] not in IMG_TYPES: continue
        img_out_path =os.path.join(pr_folder_path, img_file)
        json_out_path = img_out_path[:img_out_path.rindex('.')] + '.json'
        # 读取img的json文件载入instance对象
        try:
            json_file_path = img_file_path[:img_file_path.rindex('.')] + '.json'
            instance = json_to_instance(json_file_path)
        except Exception as e:
            instance = create_empty_json_instance(img_file_path)
        predict_boxes = img_boxes_query(img_file_path)
        print(img_file_path)
        if len(predict_boxes) == 0 and len(instance['shapes']) == 0:
            continue
        # 总待检测目标
        total_objs += len(instance['shapes'])
        # --------------------------全漏失情况--------------------------
        if recall and len(predict_boxes) == 0 and len(instance['shapes']) != 0:
            for obj in instance['shapes']:
                if obj['label'] in label_dict:
                    instance['shapes'].remove(obj)
                    continue

                if obj['label'] not in labels_tatal:
                    labels_tatal[obj['label']] = 1
                else:
                    labels_tatal[obj['label']] = labels_tatal[obj['label']] + 1
                if obj['label'] not in lou_tatal:
                    lou_tatal[obj['label']] = 1
                else:
                    lou_tatal[obj['label']] = lou_tatal[obj['label']] + 1

                obj['label'] = 'loushi_' + obj['label']
            instance_to_json(instance, json_out_path)
            shutil.copy(img_file_path, img_out_path)
            print('%s has been analyzed.' % (img_file))
            continue
        # --------------------------------------------------------------
        necessary = False
        temp = []
        # 漏失统计
        for obj in instance['shapes']:
            if obj['label'] not in labels_tatal:
                labels_tatal[obj['label']] = 1
            else:
                labels_tatal[obj['label']] = labels_tatal[obj['label']] + 1
            if obj['label'] in label_dict:
                instance['shapes'].remove(obj)
                continue
            x, y, w, h = points_to_xywh(obj)
            gt_box = Box(x, y, w, h, obj['label'])
            # 漏检 错检
            false_negative, false_label, hard = True, True, True
            for i, predict_box in enumerate(predict_boxes):
                print("i::",i)
                if gt_box.get_iou(predict_box) > iou_thres:
                    false_negative = False
                    temp.append(i)
                    if gt_box.category == predict_box.category:
                        false_label = False
                        if predict_box.confidence > hard_thres:
                            hard = False
                    else:
                        w_category = predict_box.category

            if not recall: continue
            if false_negative:
                if obj['label'] not in lou_tatal:
                    lou_tatal[obj['label']] = 1
                else:
                    lou_tatal[obj['label']] = lou_tatal[obj['label']] + 1
                obj['label'] = 'loushi_' + obj['label']
                # 总漏失
                no_detect += 1
            elif false_label:
                obj['label'] = 'cuowu_%s_' % (w_category) + obj['label']
            elif hard:
                obj['label'] = 'hard_' + obj['label']
            if false_negative or false_label or hard:
                necessary = True
        # 过检统计
        if precision:
            for i, predict_box in enumerate(predict_boxes):
                if i not in temp:
                    if predict_box.category not in guo_tatal:
                        guo_tatal[predict_box.category] = 1
                    else:
                        guo_tatal[predict_box.category] = guo_tatal[predict_box.category] + 1
                    # 总过检
                    if predict_box.confidence<guo_thres:
                        continue
                    over_detect += 1
                    obj = {'label': 'guojian_'+predict_box.category, 'shape_type': 'rectangle', 'points': [[predict_box.x, predict_box.y], [predict_box.x+predict_box.w, predict_box.y+predict_box.h]]}
                    instance['shapes'].append(obj)
                    necessary = True
        if necessary:
            instance_to_json(instance, json_out_path)
            shutil.copy(img_file_path, img_out_path)
        print('%s has been analyzed.' % (img_file))
    print('Total objects: %d, missing %d, over-detect: %d' % (total_objs, no_detect, over_detect))
    print('Total objects:',labels_tatal)
    print('lou_tatal objects:', lou_tatal)
    print('guo_tatal objects:', guo_tatal)

if __name__ == '__main__':
    from PIL import Image
    # 自定义自己的img-boxes对应方法，Box类参考utils.py
    def img_boxes_query(img_file_path):
        # cls_id_name_dict = {0:'guashang'}
        # cls_id_name_dict = {0:'aokeng', 1:'baidian', 2:'daowen', 3:'guashang', 4:'heidian', 5:'maoxu', 6:'pengshang', 7:'shahenyin',8:'tabian',9:'yise'}
        # cls_id_name_dict = {0: 'bianxing', 1: 'liewen', 2: 'luomuguoshao', 3: 'luomupengshang', 4: 'luomupengshang_nei', 5: 'pengshang', 6:'queliao', 7:'yashang', 8:'zhanliao'}
        #cls_id_name_dict = {0:'aokeng', 1:'baisezaodian', 2:'cashang', 3:'daowen', 4:'guashang', 5:'heidian', 6:'pengshang', 7:'shahenyin', 8:'tabian', 9:'yise', 10:'yiwu'}
        # cls_id_name_dict = {0:'pengshang' ,1:'guashang',2:'daowen',3:'shahenyin',4:'tabian', 5:'yise'}
        # cls_id_name_dict = {0: 'guashang', 1: 'shahenyin'}
        cls_id_name_dict = {0: 'daowen'}
        txt_folder_path = '/home/adt/project_model/yolov5-3.1/inference/output'
        txt_file = img_file_path[img_file_path.rindex(os.sep)+1:img_file_path.rindex('.')] + '.txt'
        txt_file_path = os.path.join(txt_folder_path, txt_file)
        img = Image.open(img_file_path)
        width = img.width
        height = img.height
        boxes = []
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return boxes
        # 遍历txt文件中的每一个检测目标
        for line in lines:
            words = line.split(' ')
            cls_id = int(words[0])
            cls_name = cls_id_name_dict[cls_id]
            cx, cy, w, h = float(words[2]) * width, float(words[3]) * height, float(words[4]) * width, float(words[5]) * height
            confidence = float(words[1])
            # 一个Box代表一个检测目标的xywh、label、confidence
            boxes.append(Box(cx-w/2, cy-h/2, w, h, cls_name, confidence))
        return boxes
    precision_recall_visualize(target_folder_path='/home/adt/data/data/Djian/msd/damian_dijiaodu/select_no_daowen/',
                               # 自定义的query方法
                               img_boxes_query=img_boxes_query,
                               # 保存在图片文件夹下的特定目录名
                               saved_folder_name='pr_20210514',
                               # 计算漏失
                               recall=True,
                               # 计算过检
                               precision=True)




























