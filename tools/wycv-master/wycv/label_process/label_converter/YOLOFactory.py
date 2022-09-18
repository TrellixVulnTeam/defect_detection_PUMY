import json
import multiprocessing
import os
import shutil
import time

import numpy
from tqdm import tqdm


class YOLOFactory:
    def __init__(self, label_txt, source_path, output_path, process_num):
        self.label_txt_dict = label_txt
        self.source_path = source_path
        self.output_path = output_path
        self.process_num = process_num
        self.train_list = []
        self.val_list = []
        for item in ['images', 'labels']:
            for subset in ['train', 'val']:
                os.makedirs(os.path.join(self.output_path, item, subset))

    def set_train_val(self, train_set, val_set):
        self.train_list = train_set
        self.val_list = val_set

    def save_result(self):
        # TODO: if all result divided to train_set/val_set, no need to judge witch set each item belonged to save time.
        train_img_path = os.path.join(self.output_path, 'images', 'train')
        val_img_path = os.path.join(self.output_path, 'images', 'val')
        train_label_path = os.path.join(self.output_path, 'labels', 'train')
        val_label_path = os.path.join(self.output_path, 'labels', 'val')
        result_writer_pool = multiprocessing.Pool(processes=self.process_num)
        result_writer_pbar = tqdm(total=len(self.train_list) + len(self.val_list), desc='Dividing train val set ...')
        for img_name in self.train_list:
            result_writer_pool.apply_async(self.result_writer, args=(img_name, train_img_path, train_label_path,),
                                           callback=lambda _: result_writer_pbar.update())
        for img_name in self.val_list:
            result_writer_pool.apply_async(self.result_writer, args=(img_name, val_img_path, val_label_path,),
                                           callback=lambda _: result_writer_pbar.update())
        result_writer_pool.close()
        result_writer_pool.join()

    def result_writer(self, img_name, target_img_path, target_label_path):
        if self.label_txt_dict[img_name] and self.label_txt_dict[img_name][0]:
            shutil.copy(os.path.join(self.source_path, img_name), target_img_path)
            shutil.copy(os.path.join(self.source_path, os.path.splitext(img_name)[0] + '.json'), target_img_path)
            txt_file_path = os.path.join(target_label_path, os.path.splitext(img_name)[0] + '.txt')
            with open(txt_file_path, 'w') as f:
                if len(self.label_txt_dict[img_name]) == 0:
                    f.write('')
                for line in self.label_txt_dict[img_name]:
                    f.write("%d %.03f %.03f %.03f %.03f" % tuple(line))
                    f.write('\n')

    def get_img_label_stat_dict(self):
        return {img_name: list(numpy.array(label_instance).T[0]) for img_name, label_instance in self.label_txt_dict.items()}

    def get_img_id_list(self):
        return list(self.label_txt_dict.keys())
