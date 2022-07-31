import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import numpy as np
path = "/home/zhang/datasets/PCBData/labels/"
fp = os.listdir(path)

count = 0
for i in fp:
    list = []
    count = count + 1
    f = open(path + i, "r")
    a = f.readlines()
    f = open(path + i, "w+")
    print(i)
    for jj in range(len(a)):
        new_lines = a[jj].replace("\n", "")
        list.append(new_lines)
        sp_ = list[jj].split(" ")
        dwh = 1./640
        x_c = (int(sp_[1]) + int(sp_[3]))/2.0
        y_c = (int(sp_[2]) + int(sp_[4]))/2.0
        w = int(sp_[3]) - int(sp_[1])
        h = int(sp_[4]) - int(sp_[2])
        x_c = x_c * dwh
        y_c = y_c * dwh
        w = w * dwh
        h = h * dwh

        if jj == len(a):
            new_lines = str(int(sp_[0]) - 1) + " " + str(x_c) + " " + str(y_c) + " " + str(w) + " " + str(h)
        new_lines = str(int(sp_[0]) - 1) + " " + str(x_c) + " " + str(y_c) + " " + str(w) + " " + str(h) + "\n"
        print(new_lines)
        f.write(new_lines)
    print(str(count)+"zhang\n")


    f.close()


