import os

work_dir = "/home/zhang/Dataset/"
for root, sub_folders, files in os.walk(work_dir):
    # print(root, files)
    count = 1
    for file in files:
        if file.endswith('.jpg'):
            # print(file)
            if not file.replace('.jpg', '.json') in files:
                # os.remove(os.path.join(root,file))
                print(os.path.join(root, file), count)
                count += 1

# for root, sub_folders, files in os.walk(work_dir):
#     # print(root, files)
#     count = 1
#     for file in files:
#         if file.endswith('.json'):
#             # print(file)
#             if not file.replace('.json', '.jpg') in files:
#                 # os.remove(os.path.join(root,file))
#                 print(os.path.join(root, file), count)
#                 count += 1
