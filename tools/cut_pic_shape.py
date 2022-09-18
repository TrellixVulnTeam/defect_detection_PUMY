import multiprocessing
import shutil
import os
import random
from PIL import Image
from tqdm import tqdm
import json


def cut_images(file_, path_, after_process_, size_, cut_main__):
    img = Image.open(os.path.join(path_, file_))
    width = img.size[0]
    high = img.size[1]
    name = 0
    swap = False
    if width > high:
        a = high
        high = width
        width = a
        swap = True

    for i in range(cut_main__):
        random_ = random.randint(0, int(high))
        cut_point = [int((width - size_) // 2), random_]
        if not swap:
            new_img = img.crop((cut_point[0], cut_point[1], cut_point[0] + size_, cut_point[1] + size_))
        else:
            new_img = img.crop((cut_point[1], cut_point[0], cut_point[1] + size_, cut_point[0] + size_))
        new_img.save(after_process_ + file_.replace('.jpg', '_' + str(name) + '.jpg'))
        name += 1
    img.close()


def cut_main(path_, tar, size_, cut_main_):
    origin = path_
    target = tar
    if not os.path.exists(target):
        os.mkdir(target)

    for root, sub, files in os.walk(origin):
        pbar = tqdm(total=int(len(files)))
        process_pool = multiprocessing.Pool(processes=process_num)
        for file in files:
            if file.endswith('.jpg'):
                # cut_images(file, root, target, size_, cut_main_)
                process_pool.apply_async(cut_images, args=(file,
                                                           root,
                                                           target,
                                                           size_,
                                                           cut_main_),
                                         callback=lambda _: pbar.update())
        #
        process_pool.close()
        process_pool.join()
    return target


if __name__ == '__main__':
    path = '/home/zhang/Project/Huawei_luola/good/CM_ori/'
    after_process = '/home/zhang/faiss_train/Metal/Alloy/Forge/Polish/CM/huashang&good/background/'
    label_filter = 'huashang'
    process_num = 8
    size = 224
    width = 10
    cut_num = 2
    cut_main(path, after_process, size, cut_num)
