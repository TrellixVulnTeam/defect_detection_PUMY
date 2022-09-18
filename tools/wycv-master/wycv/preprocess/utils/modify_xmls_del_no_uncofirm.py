# @Description:
# @Author     : zhangyan
# @Time       : 2020/12/30 10:45 上午
# 客户认定筛选
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time
import os


def create_Node(element, text=None):
    elem = ET.Element(element)
    elem.text = text
    return elem


def link_Node(root, element, text=None):
    """
    @param root: element的父节点
    @param element: 创建的element子节点
    @param text: element节点内容
    @return: 创建的子节点
    """
    element = create_Node(element, text)
    root.append(element)
    return element


# 保存为XML文件（美化后）
def save_XML(root, filename, indent="", newl="", encoding="utf-8"):
    rawText = ET.tostring(root)
    dom = minidom.parseString(rawText)
    with open(filename, 'w', encoding="utf-8") as f:
        dom.writexml(f, "", indent, newl, encoding)


# 修正name中带有'刮伤-边'的类别
def modify_xml(xml_path, save_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    object = root[1][0]
    for item in object.findall('item'):
        try:
            describe = item.find('describe').text
            if describe != '客户认定':
                object.remove(item)
        except:
            object.remove(item)

    rending_num = 0
    try:
        rending_num = len(root[1][0].findall('item'))
        print(rending_num)
    except:
        print('{} has no rending defects'.format(xml_path))

    if rending_num > 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = xml_path[xml_path.rindex(os.sep) + 1:]
        xml_save_path = os.path.join(save_path, save_name)
        save_XML(root, xml_save_path)


def modify_xmls(xml_path, save_path):
    t = time.time()
    xml_list = os.listdir(xml_path)
    for xml in xml_list:
        xml_file = os.path.join(xml_path, xml)  # 输入xml文件
        modify_xml(xml_file, save_path)
        print('{}'.format(xml) + ' has been modified!')
    print('解析耗时: {}'.format(time.time() - t))


if __name__ == '__main__':
    xml_path = r'D:\data\module-c\remove-tiny\damian\outputs'  # xml_folder
    save_path = r'D:\data\module-c\remove-tiny\damian\outputs-rending'  # Automatically create a save_path folder
    modify_xmls(xml_path, save_path)
