import os
import csv
import tensorflow as tf
import tarfile

from mtl.utils.log_util import get_logger
from mtl.utils.gen_tfrecords.gen_tfrecord import bytes_feature, int64_feature

g_logger = get_logger("error_check", "meta/test_infos/error_log.txt")


def create_singlelabel_tfr_example(img_name, img_data, class_id, class_label):
    tfr_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "name": bytes_feature(img_name.encode()),
                "image": bytes_feature(img_data),
                "label": int64_feature(class_id),
                "label_name": bytes_feature(class_label.encode()),
            }
        )
    )
    return tfr_example


def gen_imagenet_tfrecords(
    data_path, label_file, label_desc, data_folder, output_folder
):

    label_list = open(os.path.join(data_path, label_file)).readlines()
    label_list = [x.strip() for x in label_list]
    label2index_dict = {cat: i for i, cat in enumerate(label_list)}
    label_desc_list = open(os.path.join(data_path, label_desc)).readlines()
    label_desc_list = [x.strip() for x in label_desc_list]
    file_num = 0
    tfr_writer = None
    tf_record_path = ""
    for image_tar_file in os.listdir(os.path.join(data_path, data_folder)):
        print("Processing with: ", image_tar_file)
        if not image_tar_file.endswith(".tar"):
            continue
        if file_num % 100 == 0:
            if tfr_writer is not None:
                tfr_writer.close()
            tf_record_file = data_folder + "_" + str(file_num // 100) + ".tfrecord"
            tf_record_path = os.path.join(data_path, output_folder, tf_record_file)
            tfr_writer = tf.io.TFRecordWriter(tf_record_path)

        file_num += 1
        tar_file_path = os.path.join(data_path, data_folder, image_tar_file)
        tar = tarfile.open(tar_file_path, "r")

        num_tfr_example = 0

        tar_class_name = image_tar_file.split(".")[0]
        tar_class_id = label2index_dict[tar_class_name]
        tar_class_label = label_desc_list[tar_class_id]

        # skip the first dir member
        for member in tar.getmembers():
            img_name = member.name  # xxx.JPEG

            f = tar.extractfile(member)
            img_data = f.read()

            tfr_example = create_singlelabel_tfr_example(
                img_name, img_data, tar_class_id, tar_class_label
            )

            if tfr_example is not None:
                tfr_writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1

        print(
            "{} examples has been created for tar file {}, which are saved in {}".format(
                num_tfr_example, image_tar_file, tf_record_path
            )
        )


def gen_imagenet_val_tfrecords(
    data_path, label_file, label_desc, data_folder, output_folder
):
    label_list = open(os.path.join(data_path, label_file)).readlines()
    label_list = [x.strip() for x in label_list]
    label2index_dict = {cat: i for i, cat in enumerate(label_list)}
    label_desc_list = open(os.path.join(data_path, label_desc)).readlines()
    label_desc_list = [x.strip() for x in label_desc_list]
    file_num = 0
    tf_record_file = data_folder + ".tfrecord"
    tf_record_path = os.path.join(data_path, output_folder, tf_record_file)
    tfr_writer = tf.io.TFRecordWriter(tf_record_path)
    for sub_folder in os.listdir(os.path.join(data_path, data_folder)):
        print("Processing with: ", sub_folder)
        if sub_folder.startswith("."):
            continue

        file_num += 1
        sub_folder_path = os.path.join(data_path, data_folder, sub_folder)

        num_tfr_example = 0

        class_name = sub_folder
        class_id = label2index_dict[class_name]
        class_label = label_desc_list[class_id]

        # skip the first dir member
        for img_file in os.listdir(sub_folder_path):
            if not img_file.endswith(".JPEG"):  # xxx.JPEG
                continue

            img_data = open(os.path.join(sub_folder_path, img_file), "rb").read()

            tfr_example = create_singlelabel_tfr_example(
                img_file, img_data, class_id, class_label
            )

            if tfr_example is not None:
                tfr_writer.write(tfr_example.SerializeToString())
                num_tfr_example += 1

        print(
            "{} examples has been created, which are saved in {}".format(
                num_tfr_example, tf_record_path
            )
        )


def image1k_main():
    imagenet_path = "data/objcls-datasets/ImageNet/imagenet1k"
    label_pth = "meta/imagenet-ids.txt"
    label_description = "meta/imagenet-classes.txt"
    output_sub_folder = "tfrecords"

    # imagenet1k train tfrecords
    data_sub_folder = "train"
    gen_imagenet_tfrecords(
        imagenet_path, label_pth, label_description, data_sub_folder, output_sub_folder
    )

    # imagenet1k val tfrecords
    data_sub_folder = "val"
    gen_imagenet_val_tfrecords(
        imagenet_path, label_pth, label_description, data_sub_folder, output_sub_folder
    )


if __name__ == "__main__":
    image1k_main()
