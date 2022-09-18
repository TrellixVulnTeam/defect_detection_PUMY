import os
import shutil
import multiprocessing
from tqdm import tqdm
path = '/home/zhang/Project/Huawei/luola/Data/original/labelme'
tar = '/home/zhang/Project/Huawei/luola/Data/original/0_luola_data_labelme/cmgjzd/labelme'
count = 1
process_num = 8
# cv2.setNumThreads(0)
process_pool = multiprocessing.Pool(processes=process_num)
if not os.path.exists(tar):
    os.makedirs(tar)
    print('create',tar)

# for root, sub_floders, files in os.walk(path):
#     # pbar = tqdm(total=len(files))
#     for file in files:
#         if file.endswith('.json'):
#             process_pool.apply_async(shutil.copy, args=(os.path.join(root, file),
#                                                         os.path.join(tar, file),
#                                                         ))
#             print(os.path.join(root, file), os.path.join(tar, file), count)
#             count += 1
#     process_pool.close()
#     process_pool.join()

for root, sub_floders, files in os.walk(path):
    # pbar = tqdm(total=len(files))
    for file in files:
        process_pool.apply_async(shutil.copy, args=(os.path.join(root, file),
                                                    os.path.join(tar, file),
                                                    ))
        print(os.path.join(root, file), os.path.join(tar, file), count)
        count += 1
process_pool.close()
process_pool.join()