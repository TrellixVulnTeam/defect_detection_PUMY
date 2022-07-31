import xml.dom
import xml.dom.minidom
import os
import cv2

_TXT_PATH = '/home/zhang/datasets/pic1/Class1/Class1/Train/Label_new/label/'
_IMAGE_PATH = '/home/zhang/datasets/pic1/Class1/Class1/Train/Label_new/img/'

_INDENT = '' * 4
_NEW_LINE = '\n'
_FOLDER_NODE = ''
_ROOT_NODE = 'annotation'
_DATABASE_NAME = ''
_ANNOTATION = 'PASCAL VOC2007'
_AUTHOR = ''
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'

_IMAGE_COPY_PATH = 'JPEGImages'
_ANNOTATION_SAVE_PATH = 'Annotations'


def createElementNode(doc, tag, attr):
    element_node = doc.createElement(tag)

    text_node = doc.createTextNode(attr)

    element_node.appendChild(text_node)

    return element_node


def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)

    parent_node.appendChild(child_node)


def createObjectNode(doc, attrs):
    object_node = doc.createElement('object')

    createChildNode(doc, 'name', attrs['classification'],
                    object_node)

    createChildNode(doc, 'pose',
                    _POSE, object_node)

    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)

    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)

    bndbox_node = doc.createElement('bndbox')

    createChildNode(doc, 'xmin', attrs['x1'],
                    bndbox_node)

    createChildNode(doc, 'ymin', attrs['y1'],
                    bndbox_node)

    createChildNode(doc, 'xmax', attrs['x2'],
                    bndbox_node)

    createChildNode(doc, 'ymax', attrs['y2'],
                    bndbox_node)

    object_node.appendChild(bndbox_node)

    return object_node


def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')

    doc.writexml(tmpfile, addindent='' * 4, newl='\n', encoding='utf-8')

    tmpfile.close()

    fin = open('tmp.xml')

    fout = open(filename, 'w')

    lines = fin.readlines()

    for line in lines[1:]:

        if line.split():
            fout.writelines(line)

        # new_lines = ''.join(lines[1:])

        # fout.write(new_lines)

    fin.close()

    fout.close()


def getFileList(path):
    fileList = []
    files = os.listdir(path)
    for f in files:
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    # print len(fileList)
    return fileList


if __name__ == "__main__":

    fileList = getFileList(_TXT_PATH)
    if fileList == 0:
        os._exit(-1)

    current_dirpath = os.path.dirname(os.path.abspath('__file__'))

    if not os.path.exists(_ANNOTATION_SAVE_PATH):
        os.mkdir(_ANNOTATION_SAVE_PATH)

    if not os.path.exists(_IMAGE_COPY_PATH):
        os.mkdir(_IMAGE_COPY_PATH)

    for xText in range(len(fileList)):

        saveName = "%06d" % (xText + 1)
        pos = fileList[xText].rfind(".")
        textName = fileList[xText][:pos]

        ouput_file = open(_TXT_PATH + fileList[xText])

        lines = ouput_file.readlines()

        xml_file_name = os.path.join(_ANNOTATION_SAVE_PATH, (saveName + '.xml'))

        img = cv2.imread(os.path.join(_IMAGE_PATH, (textName + '.PNG')))

        try:
            height, width, channel = img.shape
        except AttributeError:
            continue

        print(os.path.join(_IMAGE_COPY_PATH, (textName + '.jpg')))
        cv2.imwrite(os.path.join(_IMAGE_COPY_PATH, (saveName + '.jpg')), img)
        my_dom = xml.dom.getDOMImplementation()

        doc = my_dom.createDocument(None, _ROOT_NODE, None)

        root_node = doc.documentElement

        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)

        createChildNode(doc, 'filename', saveName + '.jpg', root_node)

        source_node = doc.createElement('source')

        createChildNode(doc, 'database', _DATABASE_NAME, source_node)

        createChildNode(doc, 'annotation', _ANNOTATION, source_node)

        createChildNode(doc, 'image', 'flickr', source_node)

        createChildNode(doc, 'flickrid', 'NULL', source_node)

        root_node.appendChild(source_node)

        owner_node = doc.createElement('owner')

        createChildNode(doc, 'flickrid', 'NULL', owner_node)

        createChildNode(doc, 'name', _AUTHOR, owner_node)

        root_node.appendChild(owner_node)

        size_node = doc.createElement('size')

        createChildNode(doc, 'width', str(width), size_node)

        createChildNode(doc, 'height', str(height), size_node)

        createChildNode(doc, 'depth', str(channel), size_node)

        root_node.appendChild(size_node)

        createChildNode(doc, 'segmented', _SEGMENTED, root_node)

        for line in lines:
            s = line.rstrip('\n')

            array = s.split(' ')

            print(array)

            attrs = dict()

            attrs['x1'] = array[1]

            attrs['y1'] = array[2]

            attrs['x2'] = array[3]

            attrs['y2'] = array[4]

            attrs['classification'] = "Class" + array[0]

            print(xml_file_name)

            object_node = createObjectNode(doc, attrs)

            root_node.appendChild(object_node)

            writeXMLFile(doc, xml_file_name)
