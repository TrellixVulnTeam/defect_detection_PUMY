import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np

seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)),
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.ContrastNormalization((0.75, 1.5), per_channel=True)
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5)
])

img = Image.open('./data/img.png').convert("RGB")
# sp = img.size
img = np.asarray(img)
images_aug = seq.augment_images(img)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(images_aug)
plt.savefig('test.jpg')
pic = cv2.imread('test.jpg')
cv2.imshow('test', pic)
cv2.waitKey()
