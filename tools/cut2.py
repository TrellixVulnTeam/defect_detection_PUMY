import cv2
from matplotlib import pyplot as plt
img = cv2.imread("/home/zhang/PycharmProjects/yolov5/data/images/img.png", 0)
edges = cv2.Canny(img, 200, 300)
ret, th = cv2.threshold(img, 127, 255, 0)
# findcontours（）函数。有三个参数：输入图像、层次类型和轮廓逼近方法。
# 由函数返回的层次树相当重要：cv2.RETR_TREE参数会得到图像中轮廓的整体层次结构(contours)。如果只想得到最外面的轮廓，可以使用cv2.RETR_EXTERNAL。这对消除包含在其他轮廓中的轮廓很有用。
# 第三个参数有两种方法：
#cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1 cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 色彩空间转换函数cv2.cvtColor()
color = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
# 画出轮廓，-1,表示所有轮廓，画笔颜色为(0, 255, 0)，即Green，粗细为2
img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
cv2.imwrite("contours.png", color)
cv2.waitKey()
cv2.destroyAllWindows()