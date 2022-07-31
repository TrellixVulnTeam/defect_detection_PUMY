import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


def read_xml_annotation(root, image_id):
	in_file = open(os.path.join(root, image_id), encoding='UTF-8')
	tree = ET.parse(in_file)
	root = tree.getroot()
	bndboxlist = []

	for object in root.findall('object'):  # 找到root节点下的所有country节点
		bndbox = object.find('bndbox')  # 子节点下节点rank的值

		xmin = int(bndbox.find('xmin').text)
		xmax = int(bndbox.find('xmax').text)
		ymin = int(bndbox.find('ymin').text)
		ymax = int(bndbox.find('ymax').text)
		# print(xmin,ymin,xmax,ymax)
		bndboxlist.append([xmin, ymin, xmax, ymax])
	# print(bndboxlist)

	# ndbox = root.find('object').find('bndbox')
	return bndboxlist


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
	new_xmin = new_target[0]
	new_ymin = new_target[1]
	new_xmax = new_target[2]
	new_ymax = new_target[3]

	in_file = open(os.path.join(root, str(image_id) + '.xml'), encoding='UTF-8')  # 这里root分别由两个意思
	tree = ET.parse(in_file)
	xmlroot = tree.getroot()
	object = xmlroot.find('object')
	bndbox = object.find('bndbox')
	xmin = bndbox.find('xmin')
	xmin.text = str(new_xmin)
	ymin = bndbox.find('ymin')
	ymin.text = str(new_ymin)
	xmax = bndbox.find('xmax')
	xmax.text = str(new_xmax)
	ymax = bndbox.find('ymax')
	ymax.text = str(new_ymax)
	tree.write(os.path.join(root, str("%06d" % (str(id) + '.xml'))))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
	in_file = open(os.path.join(root, str(image_id) + '.xml'), encoding='UTF-8')  # 这里root分别由两个意思
	tree = ET.parse(in_file)
	elem = tree.find('filename')
	elem.text = id + '.jpg'
	xmlroot = tree.getroot()
	index = 0

	for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
		bndbox = object.find('bndbox')  # 子节点下节点rank的值

		# xmin = int(bndbox.find('xmin').text)
		# xmax = int(bndbox.find('xmax').text)
		# ymin = int(bndbox.find('ymin').text)
		# ymax = int(bndbox.find('ymax').text)

		new_xmin = new_target[index][0]
		new_ymin = new_target[index][1]
		new_xmax = new_target[index][2]
		new_ymax = new_target[index][3]

		xmin = bndbox.find('xmin')
		xmin.text = str(new_xmin)
		ymin = bndbox.find('ymin')
		ymin.text = str(new_ymin)
		xmax = bndbox.find('xmax')
		xmax.text = str(new_xmax)
		ymax = bndbox.find('ymax')
		ymax.text = str(new_ymax)

		index += 1

		print("index=", index)

	tree.write(os.path.join(saveroot, id + '.xml'))


def mkdir(path):
	# 去除首位空格
	path = path.strip()
	# 去除尾部 \ 符号
	path = path.rstrip("\\")
	# 判断路径是否存在
	# 存在     True
	# 不存在   False
	isExists = os.path.exists(path)
	# 判断结果
	if not isExists:
		# 如果不存在则创建目录
		# 创建目录操作函数
		os.makedirs(path)
		print(path + ' 创建成功')
		return True
	else:
		# 如果目录存在则不创建，并提示目录已存在
		print(path + ' 目录已存在')
		return False


if __name__ == "__main__":
	class_dataset = 'train'
	IMG_DIR = "new/images/" + class_dataset
	XML_DIR = "new/annotations/" + class_dataset

	AUG_XML_DIR = "new/aug/annotations/" + class_dataset  # 存储增强后的XML文件夹路径
	try:
		shutil.rmtree(AUG_XML_DIR)
	except FileNotFoundError as e:
		a = 1
	mkdir(AUG_XML_DIR)

	AUG_IMG_DIR = "new/aug/images/" + class_dataset  # 存储增强后的影像文件夹路径
	try:
		shutil.rmtree(AUG_IMG_DIR)
	except FileNotFoundError as e:
		a = 1
	mkdir(AUG_IMG_DIR)

	AUGLOOP = 4  # 每张影像增强的数量

	boxes_img_aug_list = []
	new_bndbox = []

	sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 建立lambda表达式，
	# 影像增强
	seq = iaa.Sequential([
		iaa.Fliplr(0.5),  # 镜像，50%的照片做镜像处理
		iaa.Flipud(0.5),
		iaa.ContrastNormalization((0.75, 1.5), per_channel=True),  ####0.75-1.5随机数值为alpha，对图像进行对比度增强，该alpha应用于每个通道
		iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
		#### loc 噪声均值，scale噪声方差，50%的概率，对图片进行添加白噪声并应用于每个通道
		sometimes(iaa.Crop(percent=(0, 0.1))),
		sometimes(iaa.Affine(  # 对一部分图像做仿射变换
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
			rotate=(-45, 45),  # 旋转±45度之间
			shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
			order=[0, 1],  # 使用最邻近差值或者双线性差值
			cval=(0, 255),  # 全白全黑填充
			mode=ia.ALL  # 定义填充图像外区域的方法
		)),
		# iaa.SomeOf((0, 5),
		# 		[  # 将部分图像进行超像素的表示
		# 			sometimes(
		# 				iaa.Superpixels(
		# 					p_replace=(0, 1.0),
		# 					n_segments=(20, 200)
		# 				)
		# 			),
		#
		# 			# 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
		# 				iaa.OneOf([
		# 				iaa.GaussianBlur((0, 3.0)),
		# 				iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
		# 				iaa.MedianBlur(k=(3, 11)),
		# 			]),
		#
		# 				# 锐化处理
		# 				iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
		#
		# 				# 浮雕效果
		# 				# iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
		#
		# 				# 边缘检测，将检测到的赋值0或者255然后叠在原图上
		# 				# sometimes(iaa.OneOf([
		# 				# 	iaa.EdgeDetect(alpha=(0, 0.7)),
		# 				# 	iaa.DirectedEdgeDetect(
		# 				# 	alpha=(0, 0.7), direction=(0.0, 1.0)
		# 				# 	),
		# 				# ])),
		#
		# 				# 加入高斯噪声
		# 				iaa.AdditiveGaussianNoise(
		# 					loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
		# 				),
		#
		# 				# 将1%到10%的像素设置为黑色
		# 				# 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
		# 				iaa.OneOf([
		# 					iaa.Dropout((0.01, 0.1), per_channel=0.5),
		# 					iaa.CoarseDropout(
		# 					(0.03, 0.15), size_percent=(0.02, 0.05),
		# 					per_channel=0.2
		# 					),
		# 				]),
		#
		# 				# 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
		# 				# iaa.Invert(0.05, per_channel=True),
		#
		# 				# 每个像素随机加减-10到10之间的数
		# 				iaa.Add((-10, 10), per_channel=0.5),
		#
		# 				# 像素乘上0.5或者1.5之间的数字.
		# 				iaa.Multiply((0.5, 1.5), per_channel=0.5),
		#
		# 				# 将整个图像的对比度变为原来的一半或者二倍
		# 				# iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
		#
		# 				# 将RGB变成灰度图然后乘alpha加在原图上
		# 				# iaa.Grayscale(alpha=(0.0, 1.0)),
		#
		# 				# 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
		# 				sometimes(
		# 					iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
		# 				),
		#
		# 				# 扭曲图像的局部区域
		# 				# sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
		# 		],
		#
		# 				random_order=True  # 随机的顺序把这些操作用在图像上
		# 	),
		iaa.Multiply((0.8, 1.2), per_channel=0.2),  ####20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色

	],
		random_order=True  # 随机的顺序把这些操作用在图像上
	)

	for root, sub_folders, files in os.walk(XML_DIR):

		for name in files:

			bndbox = read_xml_annotation(XML_DIR, name)
			shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
			shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.jpg'), AUG_IMG_DIR)
			print(os.path.join(IMG_DIR, name[:-4] + '.jpg'))

			for epoch in range(AUGLOOP):
				new_bndbox_list = []
				is_error = False
				seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
				# 读取图片
				img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
				sp = img.size
				img = np.asarray(img)
				# bndbox 坐标增强
				for i in range(len(bndbox)):
					bbs = ia.BoundingBoxesOnImage([
						ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
					], shape=img.shape)

					bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
					boxes_img_aug_list.append(bbs_aug)

					# new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
					n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
					n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
					n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
					n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
					if n_x1 == 1 and n_x1 == n_x2:
						print("error1", "n_x1 == 1 and n_x1 == n_x2", name)
						is_error = True
					if n_y1 == 1 and n_y2 == n_y1:
						print("error2", "n_y1 == 1 and n_y2 == n_y1", name)
						is_error = True
					if n_x1 >= n_x2 or n_y1 >= n_y2:
						print("error3", "n_x1 >= n_x2 or n_y1 >= n_y2", name)
						is_error = True
					new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])

					# 存储变化后的图片
					if not is_error:
						image_aug = seq_det.augment_images([img])[0]
						path = os.path.join(AUG_IMG_DIR, (name[:-4] + 'a' + str(epoch * 250)) + '.jpg')

						image_auged = bbs.draw_on_image(image_aug, thickness=0)

						Image.fromarray(image_auged).save(path)

				# 存储变化后的XML
				if not is_error:
					change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
												(name[:-4] + 'a' + str(epoch * 250)))
					print((name[:-4] + 'a' + str(epoch * 250)) + '.jpg')
					new_bndbox_list = []
