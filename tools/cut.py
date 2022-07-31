import cv2
import matplotlib.pyplot as plt

kernel_size = 3

plt.subplot(2,2,1)
img = cv2.imread('/home/zhang/PycharmProjects/yolov5/data/images/img.png', 0)
plt.title("gray")
plt.imshow(img)

plt.subplot(2,2,2)
edges = cv2.Canny(img, threshold1=40, threshold2=150, apertureSize=kernel_size)
plt.title("40-150")
plt.imshow(edges)

plt.subplot(2,2,3)
edges = cv2.Canny(img, threshold1=0, threshold2=150, apertureSize=kernel_size)
plt.title("0-150")
plt.imshow(edges)

plt.subplot(2,2,4)
edges = cv2.Canny(img, threshold1=40, threshold2=255, apertureSize=kernel_size)
plt.title("40-255")
plt.imshow(edges)

plt.show()
