import os
import shutil

path = '/home/zhang/Project/jingyan201/ori'
tar = '/home/zhang/Project/jingyan201'
folder_type = ['JPG', 'JSON', 'XML']
count = 1
for root, sub_floders, files in os.walk(path):
    for file in files:
        # shutil.copy(os.path.join(root, file),os.path.join(tar, file))
        # print(os.path.join(root, file), os.path.join(tar, file), count)
        # count += 1
        if file.endswith('.jpg'):
            shutil.copy(os.path.join(root, file), os.path.join(tar, folder_type[0], file))
            print(os.path.join(root, file), os.path.join(tar, folder_type[0], file), count)
            count += 1
        elif file.endswith('.json'):
            shutil.copy(os.path.join(root, file), os.path.join(tar, folder_type[1], file))
            print(os.path.join(root, file), os.path.join(tar, folder_type[1], file), count)
            count += 1
        elif file.endswith('.xml'):
            shutil.copy(os.path.join(root, file), os.path.join(tar, folder_type[2], file))
            print(os.path.join(root, file), os.path.join(tar, folder_type[2], file), count)
            count += 1
