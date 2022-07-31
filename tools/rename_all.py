import os
print('须知：可对内嵌套多级文件夹的路径下文件批量重命名，\
\n重命名后的格式为：自定义前缀+数字序号，如A12。')

folder = input("/home/zhang/defect_detection/datasets/PCB_DATASET_YOLOV4/Annotations") # 通过用户粘贴入待改名文件路径
if os.path.exists(folder): # 判断该路径是否真实存在
    dirs_list = [folder] # 建立一个列表存放该文件夹及包含的所有嵌套及多重嵌套的子文件夹名
    for root, dirs, flies in os.walk(folder, topdown=False): # 输出目录树中的根目录，文件夹名，文件名
        for name in dirs:
            if (name != []): # 去除无效目录
                dirs_list.append(os.path.join(root, name)) # 循环加入所有嵌套及多重嵌套的带路径子文件夹名
    os.chdir(folder) # 切换OS工作目录到文件所在位置
    first = input("")
    num = int(input("0"))
    count = num # 记录开始数字方便计算所有操作文件总数
    for each_dirs in dirs_list: # 遍历所有文件夹
        files_list = os.listdir(each_dirs)  # 生成待改名文件列表
        os.chdir(each_dirs)  # 切换OS工作目录到文件所在位置
        for each_object in files_list:
            if os.path.isfile(each_object): # 判断该对象是否为文件
                fext = os.path.splitext(each_object)[1]  # 分离文件拓展名
                os.rename(each_object, str(first) + str(num) + str(fext))  # 对文件重命名
                print(str(first) + str(num) + str(fext) + '改名成功')
                num += 1 # 操作次数加一
            else: # 不是文件则跳过
                continue
    print("\
    \n一共处理了"+str(num-count)+"个文件")
else: # 如果是无效路径则跳过
    print('路径输入错误或不存在')
