import os
import json


json_dir = '/home/zhang/datasets/cizhuan_seg/train_annos_new.json'  # json文件路径
out_dir = '/home/zhang/datasets/cizhuan_seg/labels/'  # 输出的 txt 文件路径


def main():
    # 读取 json 文件数据
    with open(json_dir, 'r') as load_f:
        content = json.load(load_f)
    # 循环处理
    for t in content:
        tmp = t['name'].split('.')
        filename = out_dir + tmp[0] + '.txt'

        if os.path.exists(filename):
            # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值
            x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['image_width']
            y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['image_height']
            w = (t['bbox'][2] - t['bbox'][0]) / t['image_width']
            h = (t['bbox'][3] - t['bbox'][1]) / t['image_height']
            fp = open(filename, mode="r+", encoding="utf-8")
            file_str = str(t['category'] - 1) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                       ' ' + str(round(h, 6))
            line_data = fp.readlines()

            if len(line_data) != 0:
                fp.write('\n' + file_str)
            else:
                fp.write(file_str)
            fp.close()

        # 不存在则创建文件
        else:
            fp = open(filename, mode="w", encoding="utf-8")
            fp.close()
            fp = open(filename, mode="r+", encoding="utf-8")
            line_data = fp.readlines()
            x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['image_width']
            y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['image_height']
            w = (t['bbox'][2] - t['bbox'][0]) / t['image_width']
            h = (t['bbox'][3] - t['bbox'][1]) / t['image_height']
            file_str = str(t['category'] - 1) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                       ' ' + str(round(h, 6))


            if len(line_data) != 0:
                fp.write('\n' + file_str)
            else:
                fp.write(file_str)



if __name__ == '__main__':
    main()