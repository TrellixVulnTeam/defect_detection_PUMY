import cv2
import os


def tianchong_you(img, size_w_):
    size = img.shape
    # 这里的大小可以自己设定，但是尽量是32的倍数
    constant = cv2.copyMakeBorder(img, 0, 0, 0, size_w_ - size[1], cv2.BORDER_CONSTANT, value=(107, 113, 115))  # 填充值为数据集均值
    return constant


def tianchong_xia(img, size_h_):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, size_h_ - size[0], 0, 0, cv2.BORDER_CONSTANT, value=(107, 113, 115))
    return constant


def tianchong_xy(img, size_w_, size_h_):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, size_h_ - size[0], 0, size_w_ - size[1], cv2.BORDER_CONSTANT, value=(107, 113, 115))
    return constant


def caijian(path, path_out, size_w=800, size_h=800, step_w=700, step_h=700):  # 重叠度为100
    ims_list = os.listdir(path)
    count = 0
    for im_list in ims_list:
        number = 0
        name = im_list[:-4]  # 去处“.png后缀”
        print(name)
        img = cv2.imread(path + im_list)
        size = img.shape
        if size[0] >= size_h and size[1] >= size_w:
            count = count + 1
            for h in range(0, size[0] - 1, step_h):
                start_h = h
                for w in range(0, size[1] - 1, step_w):
                    start_w = w
                    end_h = start_h + size_h
                    if end_h > size[0]:
                        start_h = size[0] - size_h
                        end_h = start_h + size_h
                    end_w = start_w + size_w
                    if end_w > size[1]:
                        start_w = size[1] - size_w
                        end_w = start_w + size_w
                    cropped = img[start_h:end_h, start_w:end_w]
                    name_img = name + '_' + str(start_h) + '_' + str(start_w)  # 用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
                    cv2.imwrite('{}/{}.png'.format(path_out, name_img), cropped)
                    number = number + 1
        if size[0] >= size_h and size[1] < size_w:
            print('图片{}需要在右面补齐'.format(name))
            count = count + 1
            img0 = tianchong_you(img, size_w)
            for h in range(0, size[0] - 1, step_h):
                start_h = h
                start_w = 0
                end_h = start_h + size_h
                if end_h > size[0]:
                    start_h = size[0] - size_h
                    end_h = start_h + size_h
                end_w = start_w + size_w
                cropped = img0[start_h:end_h, start_w:end_w]
                name_img = name + '_' + str(start_h) + '_' + str(start_w)
                cv2.imwrite('{}/{}.png'.format(path_out, name_img), cropped)
                number = number + 1
        if size[0] < size_h and size[1] >= size_w:
            count = count + 1
            print('图片{}需要在下面补齐'.format(name))
            img0 = tianchong_xia(img, size_h)
            for w in range(0, size[1] - 1, step_w):
                start_h = 0
                start_w = w
                end_w = start_w + size_w
                if end_w > size[1]:
                    start_w = size[1] - size_w
                    end_w = start_w + size_w
                end_h = start_h + size_h
                cropped = img0[start_h:end_h, start_w:end_w]
                name_img = name + '_' + str(start_h) + '_' + str(start_w)
                cv2.imwrite('{}/{}.png'.format(path_out, name_img), cropped)
                number = number + 1
        if size[0] < size_h and size[1] < size_w:
            count = count + 1
            print('图片{}需要在下面和右面补齐'.format(name))
            img0 = tianchong_xy(img, size_w, size_h)
            cropped = img0[0:size_h, 0:size_w]
            name_img = name + '_' + '0' + '_' + '0'
            cv2.imwrite('{}/{}.png'.format(path_out, name_img), cropped)
            number = number + 1
        print('图片{}切割成{}张'.format(name, number))
        print('共完成{}张图片'.format(count))


if __name__ == '__main__':
    ims_path = '/home/zhang/YOLO/images/'  # 图像数据集的路径
    path = '/home/zhang/YOLO/images/'  # 切割得到的数据集存放路径
    caijian(ims_path, path, size_w=640, size_h=640, step_h=600, step_w=600)
