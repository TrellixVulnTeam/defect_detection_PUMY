import random
from utils import *


def get_crop_num(img_size, crop_size, overlap):
    '''
    :param img_size: img长或者宽
    :param crop_size: crop的边长
    :param overlap: 相邻框的交并比
    :return: 根据overlap和crop size计算滑框截取个数
    '''
    return math.ceil((img_size-crop_size)/((1-overlap)*crop_size)) + 1

def _random_crop(cx, cy, w, h, size, shift_x_left=0.75, shift_x_right=0.25, shift_y_up=0.75, shift_y_bottom=0.25):
    '''
    :param cx: 目标中心点x
    :param cy: 目标中心点y
    :param w: 图片width
    :param h: 图片height
    :param size: 截图的size
    :param shift_x_left: 截框左边框距离cx的最左随机范围（距离像素/size）
    :param shift_x_right: 截框左边框距离cx的最右随机范围（距离像素/size）
    :param shift_y_up: 截框上边框距离cy的最上随机范围（距离像素/size）
    :param shift_y_bottom: 截框上边框距离cy的最下随机范围（距离像素/size）
    :return: 返回随机截图框
    '''
    # check crop尺寸是否超标
    size_x = w if size > w else size
    size_y = h if size > h else size
    # 截框左边框、上边框距离目标中心点的offset
    ofx, ofy = random.randint(int(size * shift_x_right), int(size * shift_x_left)), random.randint(int(size * shift_y_bottom), int(size * shift_y_up))
    cx, cy = int(cx), int(cy)
    if cy-ofy < 0:
        up, bottom = 0, size_y
    elif cy-ofy+size_y > h:
        up, bottom = h-size_y, h
    else:
        up, bottom = cy-ofy, cy-ofy+size_y
    if cx-ofx < 0:
        left, right = 0, size_x
    elif cx-ofx+size_x > w:
        left, right = w-size_x, w
    else:
        left, right = cx-ofx, cx-ofx+size_x
    return [up, bottom, left, right], [(size-size_y)//2, size-size_y-(size-size_y)//2, (size-size_x)//2, size-size_x-(size-size_x)//2]

# 根据过检和漏失的增强截图策略
def aug_crop_strategy(img, instance):
    # 截图尺寸
    size = 1536
    zhengchang=1
    # 过检增强倍数
    precision_aug = 0
    # 漏失增强倍数
    recall_aug = 0
    # 错误、难样本增强倍数
    hard_aug = 0
    crop_strategies = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    for obj in instance['shapes']:
        label = obj['label']
        cx, cy = points_to_center(obj)
        if label.startswith('guojian'):
            for i in range(precision_aug):
                crop_strategies.append(_random_crop(cx, cy, w, h, size))
        elif label.startswith('loushi'):
            for i in range(recall_aug):
                crop_strategies.append(_random_crop(cx, cy, w, h, size))
        elif label.startswith('hard') or label.startswith('cuowu'):
            for i in range(hard_aug):
                crop_strategies.append(_random_crop(cx, cy, w, h, size))
        else:
            for i in range(zhengchang):
                crop_strategies.append(_random_crop(cx, cy, w, h, size))
    return crop_strategies

def val_crop_strategy(img, instance):
    size = 512
    overlap = 0.10
    d = 100
    crop_strategies = []
    width = round(instance['imageWidth'])
    height = round(instance['imageHeight'])
    fill_size = [0, 0, 0, 0]
    if width < 1000 and height < 10000:
        num = get_crop_num(height - 2 * d, size, overlap)
        w = round(_detect_side_left_edge(img))
        for i in range(num - 1):
            crop_size = [round(i * (1 - overlap) * size) + d, round(i * (1 - overlap) * size) + size + d, w,
                         w + size]
            crop_strategies.append([crop_size, fill_size])
    elif width > 1000:
        num = get_crop_num(width, size, overlap)
        ys = []
        for obj in instance['shapes']:
            x, y, w, h = points_to_xywh(obj)
            ys.append(y + h / 2)
        ym = np.mean(ys)
        if ym - size / 2 < 0:
            ymin, ymax = 0, size
        elif ym + size / 2 > height:
            ymin, ymax = height - size, height
        else:
            ymin, ymax = round(ym - size / 2), round(ym - size / 2) + size
        for i in range(num):
            crop_size = [ymin, ymax, round(i * (1 - overlap) * size),
                         round(i * (1 - overlap) * size) + size] if i != num - 1 else [ymin, ymax, width - size,
                                                                                       width]
            crop_strategies.append([crop_size, fill_size])
    else:
        num = get_crop_num(height - 2 * d, size, overlap)
        w = round(width / 2 - size / 2 - 65)
        for i in range(num - 1):
            crop_size = [round(i * (1 - overlap) * size) + d, round(i * (1 - overlap) * size) + size + d, w,
                         w + size]
            crop_strategies.append([crop_size, fill_size])
    return crop_strategies

def my_crop_strategy(img, instance):
    size = 512
    overlap = 0.15
    crop_strategies = []
    width = round(instance['imageWidth'])
    height = round(instance['imageHeight'])
    fill_size = [0, 0, 0, 0]
    if width < 1000 and height < 10000:
        num = get_crop_num(height, size, overlap)
        w = round(_detect_side_left_edge(img))
        for i in range(num):
            crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
                         w + size] if i != num - 1 else [height - size, height, w, w + size]
            crop_strategies.append([crop_size, fill_size])
    elif width > 1000:
        num = get_crop_num(width, size, overlap)
        ys = []
        for obj in instance['shapes']:
            x, y, w, h = points_to_xywh(obj)
            ys.append(y + h / 2)
        ym = np.mean(ys)
        if ym - size / 2 < 0:
            ymin, ymax = 0, size
        elif ym + size / 2 > height:
            ymin, ymax = height - size, height
        else:
            ymin, ymax = round(ym - size / 2), round(ym - size / 2) + size
        for i in range(num):
            crop_size = [ymin, ymax, round(i * (1 - overlap) * size),
                         round(i * (1 - overlap) * size) + size] if i != num - 1 else [ymin, ymax, width - size,
                                                                                       width]
            crop_strategies.append([crop_size, fill_size])
    else:
        num = get_crop_num(height, size, overlap)
        w = round(width / 2 - size / 2)
        for i in range(num):
            crop_size = [round(i * (1 - overlap) * size), round(i * (1 - overlap) * size) + size, w,
                         w + size] if i != num - 1 else [height - size, height, w, w + size]
            crop_strategies.append([crop_size, fill_size])
    return crop_strategies

def _detect_side_left_edge(img):
    filter = np.array([[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1] for i in range(100)])
    h, w = img.shape[0], img.shape[1]
    for i in range(5, w):
        pattern = np.array(img[int(h / 2) - 50:int(h / 2) + 50, i - 5:i + 5, 0])
        if (pattern * filter).sum() / 100 >= 40 and i > 150:
            break
    return i - 50

def huawei_ac_crop_strategy(img, instance):
    size = 256
    aug = 4
    crop_strategies = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    for obj in instance['shapes']:
        cx, cy = points_to_center(obj)
        for i in range(aug):
            crop_strategies.append(_random_crop(cx, cy, w, h, size))
    return crop_strategies

# 检测特定标签的截图策略
def check_crop_strategy(img, instance):
    # 截图尺寸
    size = 1024
    # 需要查看的labels
    check_list = ['shuiyin', 'youmo']
    crop_strategies = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    for obj in instance['shapes']:
        label = obj['label']
        if label not in check_list: continue
        cx, cy = points_to_center(obj)
        crop_strategies.append(_random_crop(cx, cy, w, h, size, 0.5, 0.5, 0.5, 0.5))
    return crop_strategies

# 聚类截图策略
def clustering_crop_strategy(img, instance):
    # 截图尺寸
    size = 2048
    crop_strategies = []
    # 用来存放截取过的obj
    added = []
    w = instance['imageWidth']
    h = instance['imageHeight']
    objs = instance['shapes']
    num = len(objs)
    for i, obj in enumerate(objs):
        # 如果obj被截取过，continue
        if obj in added: continue
        # 当前聚类的外边框
        current_box = Box(*points_to_xywh(obj))
        # 开始搜寻聚类的objs
        for j in range(i+1, num):
            # 下一个obj
            next_obj = objs[j]
            # 如果下一个obj被截取过，continue
            if next_obj in added: continue
            next_box = Box(*points_to_xywh(next_obj))
            # 将下一个obj融合进当前的聚类的外边框
            combine_box = _combine_boxes(current_box, next_box)
            # 如果下一个obj不适合聚类，continue
            if combine_box.w > size or combine_box.h > size: continue
            # 聚类完成，更新当前的聚类的外边框
            current_box = combine_box
            # 将下一个obj放入added列表
            added.append(next_obj)
        if current_box.w < size and current_box.h < size:
            crop_strategies.append(_random_crop(current_box.x+current_box.w/2, current_box.y+current_box.h/2, w, h, size, (size-current_box.w/2)/size, current_box.w/2/size, (size-current_box.h/2)/size, current_box.h/2/size))
        else:
            crop_strategies.append(_random_crop(current_box.x+current_box.w/2, current_box.y+current_box.h/2, w, h, size, 0.5, 0.5, 0.5, 0.5))
    return crop_strategies

def center_random_size_crop(width,height,points,shape_type,rd):
    a=10
    if rd:
        k = random.randrange(0, a)
    else:
        k=a

    if shape_type == 'rectangle' or shape_type == 'polygon' :
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, min_y = int(min(xs)), int(min(ys))
        max_x, max_y = int(max(xs)), int(max(ys))


        if int(min_x) - k < 0:
            left_x = 0
        else:
            left_x = int(min_x) - k
        if int(max_x) + k > width:
            left_x = width - (int(max_x) - int(min_x)) - 2 * k

        return min_y, max_y, left_x,left_x + (int(max_x) - int(min_x)) + 2 * k
    elif shape_type == 'circle':
        center = [points[0][0], points[0][1]]
        radius = math.sqrt((points[1][0] - center[0]) ** 2 + (points[1][1] - center[1]) ** 2)
        min_y=int(center[1] - radius)
        max_y= int(center[1] + radius)
        min_x= int(center[0] - radius)
        max_x=int(center[0] + radius)

        if int(min_x) - k < 0:
            left_x = 0
        else:
            left_x = int(min_x) - k
        if int(max_x) + k > width:
            left_x = width - (int(max_x) - int(min_x)) - 2 * k
        return min_y, max_y, left_x,left_x + (int(max_x) - int(min_x)) + 2 * k

def random_crop(width,height,points,shape_type):
    a=10
    k = random.randrange(0, a)
    if shape_type == 'rectangle' or shape_type == 'polygon' :
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, min_y = int(min(xs)), int(min(ys))
        max_x, max_y = int(max(xs)), int(max(ys))


        if int(min_x) - k < 0:
            left_x = 0
            right_x = left_x + (int(max_x) - int(min_x)) + 2 * a
        else:
            left_x = int(min_x) - k
            right_x=left_x + (int(max_x) - int(min_x)) + 2 * a
        if int(max_x) + 2*a > width:
            left_x =  int(min_x)-k
            right_x=width
        return min_y, max_y, left_x,right_x
    elif shape_type == 'circle':
        center = [points[0][0], points[0][1]]
        radius = math.sqrt((points[1][0] - center[0]) ** 2 + (points[1][1] - center[1]) ** 2)
        min_y=int(center[1] - radius)
        max_y= int(center[1] + radius)
        min_x= int(center[0] - radius)
        max_x=int(center[0] + radius)

        if int(min_x) - k < 0:
            left_x = 0
            right_x = left_x + (int(max_x) - int(min_x)) + 2 * a
        else:
            left_x = int(min_x) - k
            right_x=left_x + (int(max_x) - int(min_x)) + 2 * a
        if int(max_x) + 2*a > width:
            left_x =  int(min_x)-k
            right_x=width
        return min_y, max_y, left_x,right_x

def random_fixed_size_crop(width,height,points,shape_type,crop_size=224,rd=False):
    # crop_size=224
    a=20
    if rd:
        k = random.randrange(-a, a)
    else:
        k=a

    if shape_type == 'rectangle' or shape_type == 'polygon' or shape_type=='linestrip'or shape_type=='line' :
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, min_y = int(min(xs)), int(min(ys))
        max_x, max_y = int(max(xs)), int(max(ys))
        center_x=int((max_x+min_x)/2)
        center_y=int((max_y+min_y)/2)


        if center_x - k-crop_size/2 < 0:
            left_x = 0
            right_x = crop_size
        else:
            left_x = center_x - k-crop_size/2
            right_x=left_x + crop_size
        if center_x + crop_size/2+k > width:
            left_x =  width-crop_size-abs(k)
            right_x=left_x+crop_size

        if center_y - k-crop_size/2 < 0:
            min_y = 0
            max_y = crop_size
        else:
            min_y = center_y - k-crop_size/2
            max_y=min_y + crop_size
        if center_y + crop_size/2+k > height:
            min_y =  height-crop_size-abs(k)
            max_y=min_y+crop_size
        return int(min_y), int(max_y), int(left_x),int(right_x)
    elif shape_type == 'circle':
        center = [points[0][0], points[0][1]]
        radius = math.sqrt((points[1][0] - center[0]) ** 2 + (points[1][1] - center[1]) ** 2)
        min_y=int(center[1] - radius)
        max_y= int(center[1] + radius)
        min_x= int(center[0] - radius)
        max_x=int(center[0] + radius)

        if int(min_x) - k < 0:
            left_x = 0
            right_x = left_x + (int(max_x) - int(min_x)) + 2 * a
        else:
            left_x = int(min_x) - k
            right_x=left_x + (int(max_x) - int(min_x)) + 2 * a
        if int(max_x) + 2*a > width:
            left_x =  int(min_x)-k
            right_x=width
        return min_y, max_y, left_x,right_x
    elif shape_type == 'point':
        center = [points[0][0], points[0][1]]
        radius = 10
        min_y=int(center[1] - radius)
        max_y= int(center[1] + radius)
        min_x= int(center[0] - radius)
        max_x=int(center[0] + radius)

        if int(min_x) - k < 0:
            left_x = 0
            right_x = left_x + (int(max_x) - int(min_x)) + 2 * a
        else:
            left_x = int(min_x) - k
            right_x=left_x + (int(max_x) - int(min_x)) + 2 * a
        if int(max_x) + 2*a > width:
            left_x =  int(min_x)-k
            right_x=width
        return min_y, max_y, left_x,right_x


def _combine_boxes(box1, box2):
    '''
    :param box1:
    :param box2:
    :return: 返回两个box的合并box
    '''
    xmin = min(box1.x, box2.x)
    ymin = min(box1.y, box2.y)
    xmax = max(box1.x+box1.w, box2.x+box2.w)
    ymax = max(box1.y+box1.h, box2.y+box2.h)
    return Box(xmin, ymin, xmax-xmin, ymax-ymin)


































