root = '/home/zhang/datasets/cizhuan_seg/'
txt_label = ['train.txt', 'test.txt', 'val.txt', ]
for txt_ in txt_label:
    f_ = open(root + 'ImageSets/' + txt_, 'r')
    save_f = open(root + txt_, 'w+')
    line = f_.readlines()
    for line_ in line:
        line_ = line_.replace('\n', '')
        save_f.write("/home/zhang/datasets/cizhuan_seg/images/" + line_ + '.jpg\n')
    f_.close()
    save_f.close()
    print(f_)
