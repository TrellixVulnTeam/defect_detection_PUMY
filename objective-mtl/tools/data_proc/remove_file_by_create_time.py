import os
import time
from multiprocessing.dummy import Pool


def remove_file(file_path, base_time):
    mtime = os.stat(file_path).st_mtime
    # print(mtime)
    file_modify_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    if file_modify_time > base_time:
        print("{0} 修改时间是: {1}".format(file_path, file_modify_time))
        os.remove(file_path)


if __name__ == "__main__":
    TARGET_DIR = "data/objcls-datasets/NormalQuality/images/normal_2400w/"
    BASE_TIME = "2021-09-28 12:00:00"

    for i in range(100):
        dir = TARGET_DIR + str(i + 1)
        pool = Pool(processes=88)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            pool.apply_async(remove_file, (file_path, BASE_TIME))

        pool.close()
        pool.join()
