import os
import cv2

PATH = '/home/zhang/cut/8.25/A1/2021-08-25/bad/CCD04/'
list_file = os.listdir('%s' % PATH)          # 该文件夹下所有的文件（包括文件夹）

for file in range(len(list_file)):
    before = cv2.imread('%s/%s' % (PATH, list_file[file]))
    after = before[270:839, 0:1280]
    if not os.path.exists('%s/cut_images/' % PATH):
        os.mkdir('%s/cut_images/' % PATH)
    cv2.imwrite('%s/cut_images/%s' % (PATH, list_file[file]), after)
    print(list_file[file])
