import os
import xlrd
import traceback
import tensorflow as tf
import numpy as np

from mtl.utils.log_util import get_logger
from mtl.utils.gen_tfrecords.gen_tfrecord import bytes_feature, int64_list_feature

g_logger = get_logger("error_check", "meta/test_infos/error_log.txt")


def create_tfr_multilabel_example(img_path, file_name, labels):
    """Create the instance with tfrecord format for the file

    :param img_path (str): the path of the image
    :param label (int): the label of the image
    :param height (int): the height of the image
    :param width (int): the width of the image
    """

    if not os.path.isfile(img_path):
        return None

    image = open(img_path, "rb").read()  # read image to memory as Bytes

    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(file_name.encode()),
                "image": bytes_feature(image),
                "label": int64_list_feature(labels),
            }
        )
    )
    return tfr_example


def gen_tfreocrd_from_excel(dataset_path, img_dir, excel_file, record_dir):
    """Generate the tfrecord for the content of file list in the dataset

    :param dataset_path: the path of the dataset
    :param img_dir: the folder of images
    :param excel_file: the name of excel file
    :param tfrecord_path: the path for storing generated tfrecord
    :return: none
    """

    train_tfrecord_path = os.path.join(dataset_path, record_dir, "train.tfrecord")
    val_tfrecord_path = os.path.join(dataset_path, record_dir, "val.tfrecord")

    excel_file = os.path.join(dataset_path, excel_file)

    book = xlrd.open_workbook(excel_file)

    sheet = book.sheet_by_name("Sheet1")
    print("Sheet info:", sheet.nrows, sheet.ncols)

    train_writer = tf.io.TFRecordWriter(train_tfrecord_path)
    val_writer = tf.io.TFRecordWriter(val_tfrecord_path)

    num_tfr_example = 0
    for i in range(1, sheet.nrows):
        try:
            img_name = sheet.row_values(i)[0]
            img_path = os.path.join(dataset_path, img_dir, img_name)
            if not os.path.exists(img_path):
                g_logger.info("Path not exists:" + img_path)
                continue

            label_vector = []
            for j in range(1, sheet.ncols):
                label_vector.append(int(sheet.row_values(i)[j]))

            tfr_example = create_tfr_multilabel_example(
                img_path, img_name, label_vector
            )

            if tfr_example is not None:

                chance = np.random.randint(100)
                if chance < 80:
                    train_writer.write(tfr_example.SerializeToString())
                else:
                    val_writer.write(tfr_example.SerializeToString())

                num_tfr_example += 1

                if num_tfr_example % 100 == 0:
                    print("Create %d TF_Example" % num_tfr_example)
            else:
                print("Error in loading: ", img_path)

        except Exception:
            g_logger.info("---------------------------------------------------------")
            g_logger.info("Exeception information:")
            g_logger.info(traceback.format_exc())
            g_logger.info("Related Path: " + img_name)
            g_logger.info("---------------------------------------------------------")

    print(
        "{} examples has been created, which are saved in {} and {}".format(
            num_tfr_example, train_tfrecord_path, val_tfrecord_path
        )
    )


if __name__ == "__main__":
    ml_dataset_path = "xxx"
    ml_img_dir = "images"
    ml_excel_file = "meta/anno.xlsx"
    ml_record_path = "tfrecords"
    gen_tfreocrd_from_excel(ml_dataset_path, ml_img_dir, ml_excel_file, ml_record_path)
