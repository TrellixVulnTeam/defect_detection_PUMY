import shutil
import os

sets = ['train', 'test', 'val']
# classes = ['broken_bottle_cap', 'bottle_cap_deformation', 'broken_edge', 'bottle_cap_spinning',
#            'cap_breakpoint', 'label_skew', 'label_wrinkle', 'label_bubble',
#            'code_normal', 'inkjet_exception']
classes = ['white-impurities', 'black-spot', 'edge-damage', 'bubble-gum']
work_dir = '/home/zhang/datasets/floor_cut/black_spot'
for classes_name in classes:
    for set_ in sets:
        if not os.path.exists(os.path.join(work_dir, 'annotations/%s' % set_)):
            os.makedirs(os.path.join(work_dir, 'annotations/%s' % set_))
        if not os.path.exists(os.path.join(work_dir, 'images/%s' % set_)):
            os.makedirs(os.path.join(work_dir, 'images/%s' % set_))
        fo1 = open(os.path.join(work_dir, 'ImageSets/%s.txt' % set_), 'r')
        lines2 = fo1.readlines()
        for file in lines2:
            file = file.replace('\n', '')
            shutil.copy(os.path.join(work_dir, 'annotations', file + '.xml'),
                        os.path.join(work_dir, 'annotations/%s' % set_, file + '.xml'))
            shutil.copy(os.path.join(work_dir, 'images', file + '.jpg'),
                        os.path.join(work_dir, 'images/%s' % set_, file + '.jpg'))
        print(lines2)
