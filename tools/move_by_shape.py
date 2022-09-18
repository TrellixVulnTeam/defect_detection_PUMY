import shutil
import os
import cv2

tar_name = ['big/', 'DM/', 'GJ/', 'ZD/']
path = '/home/zhang/datasets/bottle_cut/image/'
tar = '/home/zhang/datasets/bottle_cut/'

count = 1


def moveByShape(img_, tar_, tar_name_):
    if file.endswith('.jpg'):
        img = cv2.imread(img_)
        high = img.shape[0]
        wide = img.shape[1]
        if high >= 1000:
            os.remove(os.path.join(root, file))
            os.remove(os.path.join(root, file).replace('.jpg', '.json'))
            os.remove(os.path.join(root, file).replace('.jpg', '.xml'))
            # shutil.copy(os.path.join(root, file), tar + tar_name[0] + file)
            # shutil.copy(os.path.join(root, file).replace('.jpg', '.json'), tar + tar_name[0] + file.replace('.jpg', '.json'))
            # shutil.copy(os.path.join(root, file).replace('.jpg', '.xml'), tar + tar_name[0] + file.replace('.jpg', '.xml'))
            # print(os.path.join(root, file), tar + tar_name[0] + file)
            # print(os.path.join(root, file).replace('.jpg', '.json'), tar + tar_name[0] + file.replace('.jpg', '.json'))
            # print(os.path.join(root, file).replace('.jpg', '.xml'), tar + tar_name[0] + file.replace('.jpg', '.xml'))


for root, sub_folder, files in os.walk(path):
    for file in files:
        print(file)
        cv2.setNumThreads(0)
        moveByShape(os.path.join(root, file), tar, tar_name)
        # process_pool.apply_async(moveByShape, args=(os.path.join(root, file),
        #                                             tar,
        #                                             tar_name,))
        count += 1
