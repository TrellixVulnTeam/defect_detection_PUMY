import argparse
import copy
import glob
import json
import multiprocessing
import os.path
import yaml
from tqdm import tqdm

import wycv.preprocess.configs as configs
from wycv.preprocess.regulate import Regulate
import cv2
import wycv.preprocess.crop_lib as crop_lib
from PIL import Image


def read_complete_image(img_path):
    # with open(img_path, 'rb') as f:
    #     if img_path.lower().endswith('jpg') or img_path.lower().endswith('jpeg'):
    #         check_chars = f.read()[-2:]
    #         complete_flag = (check_chars == b'\xff\xd9')
    #     elif img_path.lower().endswith('png'):
    #         f.seek(-3, 2)
    #         check_chars = f.read()
    #         complete_flag = (check_chars == b'\x60\x82\x00' or check_chars[1:] == b'\x60\x82')
    # if not complete_flag:
    #     try:
    #         img = Image.open(img_path)
    #         img.load()
    #     except Exception as e:
    #         raise Exception('Not complete image: {}'.format(img_path))
    # else:
        return cv2.imread(img_path, 1)


class ImageCrop:
    def __init__(self, img_list, json_list, output_size, crop_params):
        self.img_list = img_list
        self.json_list = json_list
        self.crop_params = crop_params
        self.output_size = output_size if output_size is not None else self.crop_params.get('crop_size', None)
        assert self.output_size is not None, '\'crop_size\' should be given in \'crop\' params.'
        self.crop_dict_to_op = {
            'grid_crop': ImageCrop.grid_crop,
            'recursive_crop': ImageCrop.recursive_crop,
            'cluster_crop': ImageCrop.cluster_crop,
        }

    def trigger_process(self):
        try:
            crop_method = self.crop_params.get('method')
            _crop_params = copy.deepcopy(self.crop_params)
            _crop_params.pop('method')
            img_list = []
            json_list = []
            if crop_method in self.crop_dict_to_op:
                for img_item, json_item in zip(self.img_list, self.json_list):
                    img_sub_list, json_sub_list = self.crop_dict_to_op[crop_method](img_item, json_item, self.output_size, _crop_params)
                    if img_sub_list and json_sub_list:
                        img_list.extend(img_sub_list)
                        json_list.extend(json_sub_list)
            return img_list, json_list
        except Exception as e:
            raise e

    @staticmethod
    def grid_crop(img_item, json_item, output_size, crop_params):
        try:
            grid_crop_worker = crop_lib.GridCrop(img_item, json_item, output_size, crop_params)
            img_result, json_result = grid_crop_worker.trigger_crop()
            return img_result, json_result
        except Exception as e:
            raise e

    @staticmethod
    def recursive_crop(img_item, json_item, output_size, crop_params):
        try:
            recursive_crop_worker = crop_lib.RecursiveCrop(img_item, json_item, output_size, crop_params)
            img_result, json_result = recursive_crop_worker.trigger_crop()
            return img_result, json_result
        except Exception as e:
            raise e

    @staticmethod
    def cluster_crop(img_item, json_item, output_size, crop_params):
        try:
            cluster_crop_worker = crop_lib.ClusterCrop(img_item, json_item, output_size, crop_params)
            img_result, json_result = cluster_crop_worker.trigger_crop()
            return img_result, json_result
        except Exception as e:
            raise e


class ImageFilter:
    def __init__(self, img_list, json_list, output_size, filter_params):
        self.img_list = img_list
        self.json_list = json_list
        self.filter_params = filter_params

    def trigger_process(self):
        try:
            if len(self.img_list) != len(self.json_list):
                raise Exception('The length of img_list is not equal to the json_list.')
            for idx, json_item in zip(range(len(self.json_list)), self.json_list):
                if not self.filter_params.get('key'):
                    raise KeyError('\'key\' not found in {}'.format(self.filter_params))
                if self.filter_params['key'] == 'describe':
                    self.img_list[idx], self.json_list[idx] = ImageFilter.filter_describe(self.img_list[idx], json_item)
                elif self.filter_params['key'] == 'plevel':
                    self.img_list[idx], self.json_list[idx] = ImageFilter.filter_plevel(self.img_list[idx], json_item)
                elif self.filter_params['key'] == 'shape_label':
                    if not self.filter_params.get('label_filter'):
                        raise KeyError('\'label_filter\' not found in {}'.format(self.filter_params))
                    if isinstance(self.filter_params['label_filter'], list) and isinstance(self.filter_params['label_filter'][0], dict):
                        label_filter = {}
                        for filter_item in self.filter_params['label_filter']:
                            label_filter.update(filter_item)
                        self.filter_params['label_filter'] = label_filter
                    self.img_list[idx], self.json_list[idx] = ImageFilter.filter_label(self.img_list[idx], json_item,
                                                                                       self.filter_params[
                                                                                           'label_filter'])
            self.img_list = [img_item for img_item in self.img_list if img_item is not None]  # remove the None item
            self.json_list = list(filter(None, self.json_list))
            return self.img_list, self.json_list
        except Exception as e:
            raise e

    # Filter for Identified by Customer
    @staticmethod
    def filter_describe(img_item, json_item):
        try:
            for shape_item in json_item['shapes']:
                if shape_item.get('describe') and shape_item['describe'] == 'kehurending':
                    return img_item, json_item
                else:
                    return None, None
        except Exception as e:
            raise e

    # Filter by Label Name
    @staticmethod
    def filter_label(img_item, json_item, label_list):
        try:
            shapes_list = []
            labels_accept = list(filter(lambda s: s['label'] in label_list, json_item['shapes']))
            if len(labels_accept) == 0:
                return None, None
            if isinstance(label_list, dict):
                for shape_item in json_item['shapes']:
                    if shape_item['label'] in label_list:
                        shape_item['label'] = label_list[shape_item['label']]
                        shapes_list.append(shape_item)
            else:
                for shape_item in json_item['shapes']:
                    if shape_item['label'] in label_list:
                        shapes_list.append(shape_item)
            json_item['shapes'] = shapes_list
            return img_item, json_item
        except Exception as e:
            raise e

    # Filter by plevel
    @staticmethod
    def filter_plevel(img_item, json_item):
        try:
            for shape_item in json_item['shapes']:
                if shape_item.get('plevel') and shape_item['plevel'] == 'yanzhong':
                    return img_item, json_item
                else:
                    return None, None
        except Exception as e:
            raise e


class ImageReg:
    def __init__(self, img_list, json_list, output_size, reg_params):
        self.img_list = img_list
        self.json_list = json_list
        if reg_params is not None:
            self.start_point = reg_params.get('start_point')
            output_size = reg_params.get('crop_size', {})
            self.crop_w = output_size.get('width')
            self.crop_h = output_size.get('height')
            if not self.start_point or not self.crop_w or not self.crop_h:
                raise Exception('ERROR with the reg parameters')
        else:
            raise Exception('ERROR with the reg parameters')

    def trigger_process(self):
        try:
            if len(self.img_list) != len(self.json_list):
                raise Exception('The length of img_list is not equal to the json_list.')
            for idx, json_item in zip(range(len(self.json_list)), self.json_list):
                file_name = os.path.basename(json_item['imagePath'])
                json_item['imagePath'] = file_name
                regulate = Regulate(self.img_list[idx], json_item, self.start_point, self.crop_w, self.crop_h)
                self.img_list[idx], self.json_list[idx] = regulate.process()
            self.img_list = list(self.img_list)
            self.json_list = list(self.json_list)
            return self.img_list, self.json_list
        except Exception as e:
            raise e


work_key_to_op = {
    "crop": ImageCrop,
    "regulate": ImageReg,
    "filter": ImageFilter
}


class PreProcess:
    def __init__(self, configs):
        self.workflow = configs.get('workflow')
        self.work_dir = configs['work_dir']
        self.output_dir = os.path.join(configs['output_dir'], 'after_preprocess')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_size = configs.get('output_size', None)
        self.process_num = configs['process_num']

    def start_json_worker(self, json_file):
        try:
            with open(json_file) as input_file:
                json_list = [json.load(input_file)]
            img_file_path = os.path.join(self.work_dir, os.path.basename(json_list[0]['imagePath']))
            img_list = [read_complete_image(img_file_path)]
            for work_name, work_params in self.workflow.items():
                worker = work_key_to_op.get(work_name)(img_list, json_list, self.output_size, work_params)
                img_list, json_list = worker.trigger_process()
            self.save_result(img_list, json_list, self.output_dir)
        except Exception as e:
            print('Failed to process the {} or its corresponding img. ERROR: {}'.format(json_file, str(e)))

    def trigger_process(self):
        source_json_list = glob.glob(os.path.join(self.work_dir, '*.json'))
        if source_json_list:
            cv2.setNumThreads(0)
            process_pool = multiprocessing.Pool(processes=self.process_num)
            pbar = tqdm(total=len(source_json_list))
            for json_file in source_json_list:
                process_pool.apply_async(self.start_json_worker, args=(json_file, ), callback=lambda _: pbar.update())
            process_pool.close()
            process_pool.join()

    def save_result(self, img_list, json_list, output_dir):
        try:
            if len(img_list) != len(json_list):
                raise Exception('The length of img_list is not equal to the json_list.')
            for idx, json_item in zip(range(len(json_list)), json_list):
                file_name = os.path.basename(json_item['imagePath'])
                cv2.imwrite(os.path.join(output_dir, file_name), img_list[idx],
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                save_json_file = open(os.path.join(output_dir, file_name.replace('.jpg', '.json')), 'w', encoding='utf-8')
                json.dump(json_list[idx], save_json_file, indent=4)
        except Exception as e:
            raise e



def get_parser():
    parser = argparse.ArgumentParser(description='The tool used to preprocess the dataset before training.')
    parser.add_argument('--work_dir', type=str, default='', help='')
    parser.add_argument('--output_dir', type=str,
                        default='', help='')
    parser.add_argument('--image_width', type=int, default=768, help='')
    parser.add_argument('--image_height', type=int, default=768, help='')
    parser.add_argument('--width_bias', type=float, default=0.1, help='')
    parser.add_argument('--height_bias', type=float, default=0.1, help='')

    parser.add_argument('--crop_method', type=str, default='cluster_crop', help='')
    parser.add_argument('--crop_resample', type=int, default=1, help='')
    parser.add_argument('--crop_mega_instance_policy', type=str, default='center', help='')
    parser.add_argument('--filter_key', type=str, default='shape_label', help='')
    parser.add_argument('--labels_filter', type=str, default=['bianxing'], help='')
    parser.add_argument('-c', '--config', required=False, type=str, default='.yml', help='The path of the config file.')
    parser.add_argument('-p', '--process_num', type=int, default=8, help='The num of workers for multiprocess. (default: 8)')
    opt = parser.parse_args()
    config_dict = {}
    try:
        with open(opt.config) as input_file:
            config_dict = yaml.load(input_file, Loader=yaml.FullLoader)
            config_dict['process_num'] = opt.process_num
    except Exception as e:
        print('Failed to open the config file {}. ERROR: {}.'.format(opt.config, str(e)))
        print('use custom config')
        config_dict['work_dir'] = opt.work_dir
        config_dict['output_dir'] = opt.output_dir
        config_dict['output_size'] = {'width': opt.image_width, 'height': opt.image_height,
                                       'width_bias': opt.width_bias, 'height_bias': opt.height_bias}
        config_dict['workflow'] = {'crop': {'method': opt.crop_method, 'resample': opt.crop_resample, 'mega_instance_policy': opt.crop_mega_instance_policy},
                                   'filter': {'key': opt.filter_key, 'label_filter': [opt.labels_filter]}}
        config_dict['process_num'] = opt.process_num
    return config_dict


def params_check(config_dict, config_dict_model, params_error_list):
    for k in set(config_dict_model):
        if k not in config_dict:
            params_error_list.append(k)
            return False
        elif not isinstance(config_dict_model[k], dict):
            continue
        elif params_check(config_dict[k], config_dict_model[k], params_error_list):
            continue
    return True


if __name__ == '__main__':
    args = get_parser()
    error_list = []
    check_result = params_check(args, configs.config_dict_model, error_list)
    if check_result:
        preprocess = PreProcess(args)
        preprocess.trigger_process()
    else:
        raise Exception('ERROR with the config: {}'.format(str(error_list)))
