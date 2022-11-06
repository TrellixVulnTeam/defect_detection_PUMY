import os
import random
import shutil

import cv2
import albumentations as A
# import albumentations.imgaug.transforms as iaa
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


class VOCAug(object):

    def __init__(self,
                 pre_image_path=None,
                 pre_xml_path=None,
                 aug_image_save_path=None,
                 aug_xml_save_path=None,
                 start_aug_id=None,
                 labels=None,
                 max_len=4,
                 is_show=False):
        """

        :param pre_image_path:
        :param pre_xml_path:
        :param aug_image_save_path:
        :param aug_xml_save_path:
        :param start_aug_id:
        :param labels: 标签列表, 展示增强后的图片用
        :param max_len:
        :param is_show:
        """
        self.pre_image_path = pre_image_path
        self.pre_xml_path = pre_xml_path
        self.aug_image_save_path = aug_image_save_path
        self.aug_xml_save_path = aug_xml_save_path
        self.start_aug_id = start_aug_id
        self.labels = labels
        self.max_len = max_len
        self.is_show = is_show

        # print(self.labels)
        assert self.labels is not None, "labels is None!!!"

        # 数据增强选项
        # 数据增强选项
        black_block = random.randint(10, 30)
        self.aug = A.Compose([
            # # 对比度受限自适应直方图均衡
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.5),
            # # 均衡图像直方图
            # A.Equalize(by_channels=False, p=0.5),
            # # 图像均值平滑滤波
            # A.Blur(blur_limit = 7,always_apply = False,p = 0.5),
            # # 将图像行和列互换
            # A.Transpose(always_apply=False, p=0.5),
            # # 随机伽马变换
            A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, always_apply=False, p=0.5),
            # # 对图像进行光学畸变
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None,
                                mask_value=None, always_apply=False, p=0.5),
            # # 对图像进行网格失真
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None,
                             mask_value=None,
                             always_apply=False, p=0.5),
            # # 随机对图像进行弹性变换
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50,
                               interpolation=1, border_mode=4, value=None,
                               mask_value=None, always_apply=False,
                               approximate=False, p=0.5),
            # # 随机色调、饱和度、值变化
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                 val_shift_limit=20, always_apply=False, p=0.5),
            # # 填充图像
            # A.PadIfNeeded(min_height=2000, min_width=2000, border_mode=4,
            #               value=None, mask_value=None, always_apply=False,
            #               p=1.0),
            # # 运动模糊
            A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
            # # 在图像中生成正方形区域
            #
            A.Cutout(num_holes=black_block, max_h_size=black_block,
                     max_w_size=black_block,
                     fill_value=0, always_apply=False, p=0.5),
            # # 在图像上生成矩形区域
            # A.CoarseDropout(max_holes=8, max_height=8, max_width=8,
            #                 min_holes=None, min_height=None, min_width=None,
            #                 fill_value=0, always_apply=False, p=0.5),
            # # 随机更改输入图像的亮度和对比度
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                       brightness_by_max=None, always_apply=False,
                                       p=0.5),
            # # 施加摄像头传感器噪音
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5),
                       always_apply=False, p=0.5),
            # # 将图像乘以随机数或数字数组
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False,
                                  elementwise=True, always_apply=False, p=0.5),
            # # 将像素设置为0的概率
            A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0,
                           mask_drop_value=None, always_apply=False, p=0.5),
            # # 模拟图像的雾
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.08,
                        always_apply=False, p=0.5),
            # # 通过使用 2D sinc 滤波器卷积图像来创建振铃或过冲伪影
            A.RingingOvershoot(blur_limit=(7, 15), cutoff=(0.7853981633974483, 1.5707963267948966),
                               always_apply=False, p=0.5),
            # # 锐化输入图像并将结果与原始图像叠加
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            #
            A.OneOf([
                # 水平翻转、旋转、垂直翻转
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.VerticalFlip(p=0.5)
            ], p=1),
            A.OneOf([
                # 模糊、失真、噪声

                A.GaussianBlur(p=0.5),
                A.OpticalDistortion(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.5)
            ], p=1),
            A.OneOf([
                # RGB平移
                A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
                # 随机排列通道
                A.ChannelShuffle(p=0.3),  # 随机排列通道
                # 随机改变图像的亮度、对比度、饱和度、色调
                A.ColorJitter(p=0.3),
                # 随机丢弃通道
                A.ChannelDropout(p=0.3),
            ], p=0.5),
            A.ColorJitter(p=0.3),
            # A.Downscale(p=0.5),  # 随机缩小和放大来降低图像质量
            A.Emboss(p=0.2),  # 压印输入图像并将结果与原始图像叠加
        ],
            A.BboxParams(format='pascal_voc', min_area=0., min_visibility=0., label_fields=['category_id'])
        )
        print('--------------*--------------')
        print("labels: ", self.labels)
        # if self.start_aug_id is None:
        #     self.start_aug_id = len(os.listdir(self.pre_xml_path))
        #     print("the start_aug_id is not set, default: len(images)", self.start_aug_id)
        # print('--------------*--------------')

    def get_xml_data(self, xml_filename):
        with open(os.path.join(self.pre_xml_path, xml_filename), 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            shuffix = tree.find('filename').text.split('.')[-1]
            # image_name = tree.find('filename').text
            image_name = xml_filename.replace('.xml', '.' + shuffix)
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            bboxes = []
            cls_id_list = []
            for obj in root.iter('object'):
                # difficult = obj.find('difficult').text
                difficult = obj.find('difficult').text
                cls_name = obj.find('name').text  # label
                # if cls_name not in LABELS or int(difficult) == 1:
                if cls_name not in LABELS:
                    continue
                xml_box = obj.find('bndbox')

                xmin = int(xml_box.find('xmin').text)
                ymin = int(xml_box.find('ymin').text)
                xmax = int(xml_box.find('xmax').text)
                ymax = int(xml_box.find('ymax').text)

                # 标注越界修正
                if xmax > w:
                    xmax = w
                if ymax > h:
                    ymax = h
                bbox = [xmin, ymin, xmax, ymax]
                bboxes.append(bbox)
                cls_id_list.append(self.labels.index(cls_name))

            # 读取图片
            # image = cv2.imread(os.path.join(self.pre_image_path, image_name))
            image = cv2.cvtColor(np.asarray(Image.open(os.path.join(self.pre_image_path, image_name))),
                                 cv2.COLOR_RGB2BGR)
        return bboxes, cls_id_list, image, image_name

    def aug_image(self, AUGLOOP):
        xml_list = os.listdir(self.pre_xml_path)

        # cnt = self.start_aug_id
        for xml in xml_list:
            file_suffix = xml.split('.')[-1]
            if file_suffix not in ['xml']:
                continue

            bboxes, cls_id_list, image, image_name = self.get_xml_data(xml)

            anno_dict = {'image': image, 'bboxes': bboxes, 'category_id': cls_id_list}
            # 获得增强后的数据 {"image", "bboxes", "category_id"}
            shutil.copy(os.path.join(self.pre_xml_path, xml),
                        os.path.join(self.aug_xml_save_path, xml))
            shutil.copy(os.path.join(self.pre_image_path, image_name),
                        os.path.join(self.aug_image_save_path, image_name))
            record_aug = []
            for epoch in range(AUGLOOP):
                augmented = self.aug(**anno_dict)
                # 保存增强后的数据
                flag = self.save_aug_data(augmented, image_name, epoch, record_aug)
            print(image_name, '增强了', len(record_aug), '倍')
            # if flag:
            #     cnt += 1
            # else:
            #     continue

    def save_aug_data(self, augmented, image_name, cnt, record_aug):
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_category_id = augmented['category_id']
        # print(aug_bboxes)
        # print(aug_category_id)

        # name = '0' * self.max_len
        # 获取图片的后缀名
        image_suffix = image_name.split(".")[-1]

        # 未增强对应的xml文件名
        pre_xml_name = image_name.replace(image_suffix, 'xml')

        # 获取新的增强图像的文件名
        # cnt_str = str(cnt)
        # length = len(cnt_str)
        # new_image_name = name[:-length] + cnt_str + "." + image_suffix
        new_image_name = str(cnt) + '_' + image_name
        # 获取新的增强xml文本的文件名
        new_xml_name = new_image_name.replace(image_suffix, 'xml')

        # 获取增强后的图片新的宽和高
        new_image_height, new_image_width = aug_image.shape[:2]

        # 深拷贝图片
        aug_image_copy = aug_image.copy()

        # 在对应的原始xml上进行修改, 获得增强后的xml文本
        with open(os.path.join(self.pre_xml_path, pre_xml_name), 'r') as pre_xml:
            aug_tree = ET.parse(pre_xml)

        # 修改image_filename值
        root = aug_tree.getroot()
        aug_tree.find('filename').text = new_image_name

        # 修改变换后的图片大小
        size = root.find('size')
        size.find('width').text = str(new_image_width)
        size.find('height').text = str(new_image_height)

        # 修改每一个标注框
        for index, obj in enumerate(root.iter('object')):
            # obj.find('name').text = self.labels[aug_category_id[index]]
            try:
                xmin, ymin, xmax, ymax = aug_bboxes[index]
            except Exception as e:
                print('xmin, ymin, xmax, ymax = aug_bboxes[index]', e)
                return False
            xml_box = obj.find('bndbox')
            xml_box.find('xmin').text = str(int(xmin))
            xml_box.find('ymin').text = str(int(ymin))
            xml_box.find('xmax').text = str(int(xmax))
            xml_box.find('ymax').text = str(int(ymax))
            if self.is_show:
                tl = 2
                text = f"{LABELS[aug_category_id[index]]}"
                t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tl)[0]
                cv2.rectangle(aug_image, (int(xmin), int(ymin) - 3),
                              (int(xmin) + t_size[0], int(ymin) - t_size[1] - 3),
                              (0, 0, 255), -1, cv2.LINE_AA)  # filled
                cv2.putText(aug_image, text, (int(xmin), int(ymin) - 2), 0, tl / 3, (255, 255, 255), tl,
                            cv2.LINE_AA)
                cv2.rectangle(aug_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)

        if self.is_show:
            # aug_image = cv2.resize(aug_image, (768, 768))
            cv2.imshow('aug_image_show', aug_image)
            # 按下s键保存增强,否则取消保存此次增强
            key = cv2.waitKey(0)
            if key & 0xff == ord('s'):
                pass
            else:
                return False
        # 保存增强后的图片
        cv2.imwrite(os.path.join(self.aug_image_save_path, new_image_name), aug_image_copy)
        # 保存增强后的xml文件
        tree = ET.ElementTree(root)
        tree.write(os.path.join(self.aug_xml_save_path, new_xml_name))
        record_aug.append(new_image_name)
        return True


if __name__ == '__main__':
    # 原始的xml路径和图片路径
    PRE_IMAGE_PATH = '/home/zhang/datasets/bottle_cut/source_cap/train'
    PRE_XML_PATH = '/home/zhang/datasets/bottle_cut/source_cap/train'

    # 增强后保存的xml路径和图片路径
    AUG_SAVE_IMAGE_PATH = '/home/zhang/datasets/bottle_cut_aug/5/source/train'
    AUG_SAVE_XML_PATH = '/home/zhang/datasets/bottle_cut_aug/5/source/train'
    if not os.path.exists(AUG_SAVE_IMAGE_PATH):
        os.makedirs(AUG_SAVE_IMAGE_PATH, exist_ok=True)
    if not os.path.exists(AUG_SAVE_XML_PATH):
        os.makedirs(AUG_SAVE_XML_PATH, exist_ok=True)
    # 标签列表
    # LABELS = ['white-impurities', 'black-spot', 'edge-damage', 'bubble-gum']
    LABELS = ['broken_bottle_cap', 'bottle_cap_deformation', 'broken_edge',
              'bottle_cap_spinning', 'cap_breakpoint', 'code_normal',
              'inkjet_exception']
    AUGLOOP = 5
    aug = VOCAug(
        pre_image_path=PRE_IMAGE_PATH,
        pre_xml_path=PRE_XML_PATH,
        aug_image_save_path=AUG_SAVE_IMAGE_PATH,
        aug_xml_save_path=AUG_SAVE_XML_PATH,
        start_aug_id=None,
        labels=LABELS,
        is_show=False,
    )

    aug.aug_image(AUGLOOP)

    # cv2.destroyAllWindows()
