# -*- coding:utf-8 -*-

import cv2

img = cv2.imread('/home/zhang/cut/8.25/A1/2021-08-25/bad/CCD04/20210825-103316423-0104000-000018.bmp')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def mouse_click(event, x, y, flags, para):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        print('PIX:', y, x)
        print("BGR:", img[y, x])
        print("GRAY:", grey[y, x])
        print("HSV:", hsv[y, x])


if __name__ == '__main__':
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_click)
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()
