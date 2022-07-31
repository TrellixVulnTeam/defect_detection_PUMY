import xml.etree.ElementTree as ET
from os import getcwd

sets = [('myData', 'train'), ('myData', 'val'), ('myData', 'test')]
# sets = [('myData', 'train')]
classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def convert_annotation(year, image_id, list_file):
    ##in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open('/home/zhang/datasets/PCBData_Color_Yolov4/Annotations/%s.xml' % (image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    wd = getcwd()

    for year, image_set in sets:
        image_ids = open('/home/zhang/datasets/PCBData_Color_Yolov4/ImageSets/%s.txt' % (image_set)).read().strip().split()
        list_file = open('%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('/home/zhang/datasets/PCBData_Color_Yolov4/images/%s.jpg' % (image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
