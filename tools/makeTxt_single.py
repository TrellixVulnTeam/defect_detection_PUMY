import os
import random
import shutil

# trainval_percent = 1
# train_percent = 1
work_dir = '/home/zhang/datasets/floor_cut_blance/480'
sets=['train', 'test', 'val']
classes = ['white-impurities', 'black-spot', 'edge-damage', 'bubble-gum']
for set_name in sets:
    xmlfilepath = os.path.join(work_dir, 'Annotations/%s' % set_name)
    txtsavepath = os.path.join(work_dir, 'ImageSets')
    if not os.path.exists(txtsavepath):
        os.mkdir(txtsavepath)
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)

    train_list = random.sample(list, num)

    fo = open(txtsavepath + '/' + '%s.txt' % set_name, 'w')
    for i in train_list:
        name = os.path.splitext(total_xml[i])[0] + '\n'
        fo.write(name)
    fo.close()

