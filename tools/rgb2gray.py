import cv2 as cv
import numpy as np
import os
img_cat_gray = cv.imread("/home/zhang/defect_detection/tools/000000000019.jpg", 0)
img_cat_color = cv.imread("/home/zhang/defect_detection/tools/000000000019.jpg")

cv.imwrite("ceshi.png",img_cat_gray)
img_cat_gray = cv.imread("ceshi.png")
print("img_cat_gray: ", img_cat_gray.shape)
print("img_cat_color: ", img_cat_color.shape)