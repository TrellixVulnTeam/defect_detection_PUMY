# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 22:00
# @Author  : zhiming.qian
# @Email   : zhiming.qian@micro-i.com.cn
# @File    : gen_split_dataset.py

import os
import numpy as np


def create_filename_lists(
    dataset_path, img_dir_name, label_dir_name, val_percentage=10, test_percentage=0
):
    """
    Create data lists with all dictionaries in the data path.
    :param dataset_path: the path of dataset
    :param img_dir_name: the folder name for images
    :param label_dir_name: the folder name for labels
    :param val_percentage: the percentage of validation data
    :param test_percentage: the percentage of testing data
    :return: the dict of train, val and test lists of file names
    """
    result = {}
    train_file_list = []
    val_file_list = []
    test_file_list = []
    img_path = os.path.join(dataset_path, img_dir_name)
    label_path = os.path.join(dataset_path, label_dir_name)

    for file_name in os.listdir(img_path):
        # achieve the label files
        if not os.path.isfile(os.path.join(label_path, file_name[:-4] + ".txt")):
            continue

        # only store the file names
        chance = np.random.randint(100)
        if chance < 100 - val_percentage - test_percentage:
            train_file_list.append(file_name[:-4])
        elif chance < 100 - test_percentage:
            val_file_list.append(file_name[:-4])
        else:
            test_file_list.append(file_name[:-4])

    result["train"] = train_file_list
    result["val"] = val_file_list
    result["test"] = test_file_list

    return result
