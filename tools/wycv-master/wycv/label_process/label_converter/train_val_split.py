import math
import numpy as np
import random


def random_split(total_img_id_list, ratio, random_seed):
    n_total = len(total_img_id_list)
    offset = math.ceil(n_total * ratio)
    if n_total == 0 or offset < 1:
        return total_img_id_list, []
    elif offset >= n_total:
        return [], total_img_id_list
    try:
        random.Random(random_seed).shuffle(total_img_id_list)
    except:
        random.shuffle(total_img_id_list)
    val_sublist = total_img_id_list[:offset]
    train_sublist = total_img_id_list[offset:]
    return train_sublist, val_sublist


def filter_split(total_img_id_list, ratio, data_instance, filter_label, level):
    n_total = len(total_img_id_list)
    val_sum = math.ceil(len(total_img_id_list) * ratio)
    if n_total == 0 or val_sum < 1:
        return total_img_id_list, []
    elif val_sum >= n_total:
        return [], total_img_id_list

    # statistic cate_id by each img
    img_label_stat_dict = data_instance.get_img_label_stat_dict()

    # divide to three level and sort
    img_split_by_level = [[], [], []]
    for img_id, img_label_item in img_label_stat_dict.items():
        intersection_set = set(img_label_item).intersection(set(filter_label))
        if len(intersection_set) == len(set(filter_label)) == len(set(img_label_item)):
            img_split_by_level[0].append(img_id)
        elif len(set(img_label_item)) == len(intersection_set) < len(set(filter_label)):
            img_split_by_level[1].append((img_id, len(intersection_set)))
        elif len(intersection_set) > 0:
            img_split_by_level[2].append((img_id, len(intersection_set)))
    for idx, level_item in enumerate(img_split_by_level[1:]):
        img_split_by_level[idx + 1] = [item[0] for item in sorted(level_item, key=lambda x: x[1], reverse=True)]

    # divide each level with ratio
    img_num_by_level = [len(level_item) for level_item in img_split_by_level]
    split_base_num = np.sum(img_num_by_level[:level])
    train_img_by_level, val_img_by_level = [], []
    if split_base_num / val_sum < ratio:
        raise Exception('The total num of images under level-{} and higher is less than the ratio * total num, '
                        'please give a smaller ratio or bigger level.'.format(level))
    val_remain = val_sum
    for idx in range(level):
        if idx == 0 or idx <= level - 2:
            val_img_num = int(min(val_sum * img_num_by_level[idx] / split_base_num, img_num_by_level[idx]))
        else:
            val_img_num = val_remain
        random.shuffle(img_split_by_level[idx])
        if img_split_by_level[idx]:
            val_img_by_level.extend(img_split_by_level[idx][:val_img_num])
            train_img_by_level.extend(img_split_by_level[idx][val_img_num:])
        val_remain -= val_img_num
    return train_img_by_level, val_img_by_level
