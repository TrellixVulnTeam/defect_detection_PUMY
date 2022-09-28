import json
import multiprocessing
import os
import shutil
import sys
import copy

import cv2
import PIL.Image

import numpy
from tqdm import tqdm


def lbl_save(filename, lbl):
    import imgviz
    if os.path.splitext(filename)[1] != ".png":
        filename += ".png"
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(numpy.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )


def img_id_process(instance_item):
    instance_copy = instance_item.copy()
    instance_copy['id'] += 1
    return instance_copy


def anno_id_process(instance_item, reorder):
    instance_copy = instance_item.copy()
    instance_copy['id'] += 1
    if reorder:
        instance_copy['category_id'] += 1
    try:
        instance_copy['image_id'] += 1
    except:
        pass
    return instance_copy


class COCOFactory:
    def __init__(self, image_list, source_path, output_path, gen_mask_flag, color_mask_flag, reorder_flag, process_num):
        self.image_list = image_list
        self.source_path = source_path
        self.output_path = output_path
        self.train_path = os.path.join(output_path, 'train2017')
        self.val_path = os.path.join(output_path, 'val2017')
        self.anno_path = os.path.join(output_path, 'annotations')
        self.reorder_flag = reorder_flag
        self.process_num = process_num
        self.category_list = []
        self.annotation_list = multiprocessing.Manager().list()
        self.img_mask_dict = {}
        self.train_list = []
        self.val_list = []
        self.gen_mask_flag = gen_mask_flag
        self.color_mask_flag = color_mask_flag

        os.makedirs(self.train_path)
        os.makedirs(self.val_path)
        os.makedirs(self.anno_path)

    def init_category_list(self, category_dict):
        self.category_list = [{'supercategory': k, 'id': v+1, 'name': k} for k, v in category_dict.items()] \
            if self.reorder_flag else [{'supercategory': k, 'id': v, 'name': k} for k, v in category_dict.items()]

    def add_anno(self, anno_item):
        self.annotation_list.append(anno_item)

    def set_train_val(self, train_set, val_set, test_set):
        self.train_list = train_set
        self.val_list = val_set
        self.test_list = test_set

    def save_result(self):
        annotation_list = list(self.annotation_list)
        # TODO: if all result divided to train_set/val_set, save directly to save time.
        train_json = {'images': multiprocessing.Manager().list(), 'annotations': multiprocessing.Manager().list(), 'categories': self.category_list}
        val_json = {'images': multiprocessing.Manager().list(), 'annotations': multiprocessing.Manager().list(), 'categories': self.category_list}
        test_json = {'images': multiprocessing.Manager().list(), 'annotations': multiprocessing.Manager().list(), 'categories': self.category_list}
        save_pbar = tqdm(total=len(self.image_list) + len(annotation_list), desc='Dividing train val set ...')
        img_process_pool = multiprocessing.Pool(processes=self.process_num)
        for img_item in self.image_list:
            img_process_pool.apply_async(self.img_process, args=(img_item, train_json, val_json, test_json,),
                                         callback=lambda _: save_pbar.update())
        img_process_pool.close()
        img_process_pool.join()

        anno_process_pool = multiprocessing.Pool(processes=self.process_num)
        for anno_item in annotation_list:
            anno_process_pool.apply_async(self.anno_process, args=(anno_item, train_json, val_json, test_json,),
                                          callback=lambda _: save_pbar.update())
        anno_process_pool.close()
        anno_process_pool.join()

        for json_item in [train_json, val_json, test_json]:
            json_item['images'] = list(json_item['images'])
            json_item['annotations'] = list(json_item['annotations'])

        with open(os.path.join(self.anno_path, 'instances_train2017.json'), 'w', encoding='utf-8') as out_file:
            json.dump(train_json, out_file, ensure_ascii=False, indent=2)
        with open(os.path.join(self.anno_path, 'instances_val2017.json'), 'w', encoding='utf-8') as out_file:
            json.dump(val_json, out_file, ensure_ascii=False, indent=2)
        with open(os.path.join(self.anno_path, 'instances_test2017.json'), 'w', encoding='utf-8') as out_file:
            json.dump(test_json, out_file, ensure_ascii=False, indent=2)

        if self.gen_mask_flag:
            self.gen_mask()

    def img_process(self, img_item, train_json, val_json, test_json):
        if img_item['id'] in self.train_list:
            train_json['images'].append(img_id_process(img_item))
            shutil.copy(os.path.join(self.source_path, img_item['file_name']), self.train_path)
            shutil.copy(os.path.join(self.source_path, os.path.splitext(img_item['file_name'])[0] + '.json'), self.train_path)
        elif img_item['id'] in self.val_list:
            val_json['images'].append(img_id_process(img_item))
            shutil.copy(os.path.join(self.source_path, img_item['file_name']), self.val_path)
            shutil.copy(os.path.join(self.source_path, os.path.splitext(img_item['file_name'])[0] + '.json'), self.val_path)
        elif img_item['id'] in self.test_list:
            test_json['images'].append(img_id_process(img_item))
            shutil.copy(os.path.join(self.source_path, img_item['file_name']), self.test_path)
            shutil.copy(os.path.join(self.source_path, os.path.splitext(img_item['file_name'])[0] + '.json'), self.test_path)

    def anno_process(self, anno_item, train_json, val_json, test_json):
        if anno_item['image_id'] in self.train_list:
            train_json['annotations'].append(anno_id_process(anno_item, self.reorder_flag))
        elif anno_item['image_id'] in self.val_list:
            val_json['annotations'].append(anno_id_process(anno_item, self.reorder_flag))
        elif anno_item['image_id'] in self.test_list:
            test_json['annotations'].append(anno_id_process(anno_item, self.reorder_flag))

    def gen_mask(self):
        for item in ['train2017', 'val2017', 'test2017']:
            stuffthingmaps_path = os.path.join(self.output_path, 'stuffthingmaps', item)
            if self.color_mask_flag:
                os.makedirs(os.path.join(stuffthingmaps_path, 'mask'))
            mask_grey_path = os.path.join(stuffthingmaps_path, 'labels')
            os.makedirs(mask_grey_path)
        mask_except_path = os.path.join(self.output_path, 'except')
        os.makedirs(mask_except_path)
        gen_mask_pool = multiprocessing.Pool(processes=self.process_num)
        gen_mask_pbar = tqdm(total=len(self.image_list), desc='Generating the masks ...')
        for img_item in self.image_list:
            gen_mask_pool.apply_async(self.mask_generator, args=(img_item, mask_except_path,),
                                      callback=lambda _: gen_mask_pbar.update())
        gen_mask_pool.close()
        gen_mask_pool.join()

    def mask_generator(self, img_item, mask_except_path):
        cv2.setNumThreads(0)
        try:
            if img_item['id'] in self.train_list:
                mask_rgb_path = os.path.join(self.output_path, 'stuffthingmaps', 'train2017', 'mask')
                mask_grey_path = os.path.join(self.output_path, 'stuffthingmaps', 'train2017', 'labels')
            elif img_item['id'] in self.val_list:
                mask_rgb_path = os.path.join(self.output_path, 'stuffthingmaps', 'val2017', 'mask')
                mask_grey_path = os.path.join(self.output_path, 'stuffthingmaps', 'val2017', 'labels')
            elif img_item['id'] in self.test_list:
                mask_rgb_path = os.path.join(self.output_path, 'stuffthingmaps', 'test2017', 'mask')
                mask_grey_path = os.path.join(self.output_path, 'stuffthingmaps', 'test2017', 'labels')
            cls = numpy.zeros((img_item['height'], img_item['width']), dtype=numpy.int32)
            ins = numpy.zeros_like(cls)
            for mask_idx, mask_item in enumerate(list(self.img_mask_dict[img_item['id']])):
                ins_id = mask_idx + 1
                cls_id = mask_item['category_id'] + 1
                cls[mask_item['mask']] = cls_id
                ins[mask_item['mask']] = ins_id
            if self.color_mask_flag:
                lbl_save(os.path.join(mask_rgb_path, os.path.splitext(img_item['file_name'])[0]), cls)
            opencv_img_label = numpy.asarray(cls)
            cv2.imwrite(os.path.join(mask_grey_path, os.path.splitext(img_item['file_name'])[0]) + '.png', opencv_img_label,
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
        except Exception as e:
            shutil.copy(os.path.join(self.source_path, os.path.splitext(img_item['file_name'])[0]+'.json'), mask_except_path)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('ERROR: {}'.format(str(e)))

    def get_img_label_stat_dict(self):
        img_label_stat_dict = {}
        for label_instance in list(self.annotation_list):
            try:
                img_label_stat_dict[label_instance['image_id']].append(label_instance['category_id'])
            except:
                img_label_stat_dict[label_instance['image_id']] = [label_instance['category_id']]
        return img_label_stat_dict

    def get_img_id_list(self):
        return list(self.img_mask_dict.keys())
