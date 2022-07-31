# -*- coding: utf-8 -*-
# @Function:There are 20 classes in VOC data set. If you need to extract specific classes, you can use this program to extract them.

import os
import shutil
dataset_name = 'floor'
classes = ['white-impurities','black-spot','edge-damage','bubble-gum']
# classes = ['broken_bottle_cap', 'bottle_cap_deformation', 'broken_edge', 'bottle_cap_spinning',
#            'cap_breakpoint', 'label_skew', 'label_wrinkle', 'label_bubble',
#            'code_normal', 'inkjet_exception']
# class_used = 'damage'
for class_used in classes:
    ann_filepath = './'+ dataset_name +'/Annotations/'
    img_filepath = './'+ dataset_name +'/images/'
    img_savepath = './'+ dataset_name +'_aug/' + class_used + '/images/'
    ann_savepath = './'+ dataset_name +'_aug/' + class_used + '/Annotations/'
    if not os.path.exists('./'+ dataset_name +'_aug/' + class_used):
        os.mkdir('./'+ dataset_name +'_aug/' + class_used)
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)

    if not os.path.exists(ann_savepath):
        os.mkdir(ann_savepath)
    names = locals()

    for file in os.listdir(ann_filepath):
        print(file)
        fp = open(ann_filepath  + file)
        ann_savefile = ann_savepath + file
        fp_w = open(ann_savefile, 'w')
        lines = fp.readlines()

        ind_start = []
        ind_end = []
        lines_id_start = lines[:]
        lines_id_end = lines[:]

        classes1 = '        <name>' + class_used + '</name>\n'
        # 在xml中找到object块，并将其记录下来
        while '    <object>\n' in lines_id_start:
            a = lines_id_start.index('    <object>\n')
            ind_start.append(a)
            lines_id_start[a] = "delete"

        while '    </object>\n' in lines_id_end:
            b = lines_id_end.index('    </object>\n')
            ind_end.append(b)
            lines_id_end[b] = "delete"

        # names中存放所有的object块
        i = 0
        for k in range(0, len(ind_start)):
            names['block%d' % k] = []
            for j in range(0, len(classes)):
                if classes[j] in lines[ind_start[i] + 1]:
                    a = ind_start[i]
                    for o in range(ind_end[i] - ind_start[i] + 1):
                        names['block%d' % k].append(lines[a + o])
                    break
            i += 1
            # print(names['block%d' % k])

        # xml头
        string_start = lines[0:ind_start[0]]
        # xml尾
        string_end = [lines[len(lines) - 1]]

        # 在给定的类中搜索，若存在则，写入object块信息
        a = 0
        for k in range(0, len(ind_start)):
            if classes1 in names['block%d' % k]:
                a += 1
                string_start += names['block%d' % k]
        string_start += string_end
        for c in range(0, len(string_start)):
            fp_w.write(string_start[c])
        fp_w.close()
        # 如果没有我们寻找的模块，则删除此xml，有的话拷贝图片
        if a == 0:
            os.remove(ann_savepath + file)
        else:
            name_img = img_filepath + os.path.splitext(file)[0] + ".bmp"
            shutil.copy(name_img, img_savepath)
        fp.close()