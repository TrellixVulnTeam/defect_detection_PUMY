import os
import cv2
import  xml.dom.minidom
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='对图片进行可视化')
    parser.add_argument('--img', help='图片路径',required=True,
                        type=str)
    parser.add_argument('--xml', help='标签路径', required=True,
                        type=str)
    parser.add_argument('--save_path', help='保存路径', required=True,
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    image_path=args.img
    annotation_path=args.xml
    save_path =args.save_path

    img_names = os.listdir(image_path)
    xml_names = os.listdir(annotation_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, "vis_img")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for img_n, xml_n in tqdm(zip(img_names, xml_names)):
        img_path =os.path.join(image_path, img_n)
        xml_path =os.path.join(annotation_path,xml_n)
        img = cv2.imread(img_path)
        if img is None:
            pass
        try:
            dom = xml.dom.minidom.parse(xml_path)
        except:
            continue
        root = dom.documentElement
        objects=dom.getElementsByTagName("object")
        for object in objects:
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            ymin = bndbox.getElementsByTagName('ymin')[0]
            xmax = bndbox.getElementsByTagName('xmax')[0]
            ymax = bndbox.getElementsByTagName('ymax')[0]
            xmin_data=int(float(xmin.childNodes[0].data))
            ymin_data=int(float(ymin.childNodes[0].data))
            xmax_data=int(float(xmax.childNodes[0].data))
            ymax_data=int(float(ymax.childNodes[0].data))
            label_name=object.getElementsByTagName('name')[0].childNodes[0].data
            cv2.rectangle(img,(xmin_data,ymin_data),(xmax_data,ymax_data),(55,255,155),1)
            cv2.putText(img,label_name,(xmin_data,ymin_data),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
        flag=0
        flag=cv2.imwrite(os.path.join(save_path,img_n),img)
        if not (flag):
            print(img_n,"error")
    print("all done ====================================")
