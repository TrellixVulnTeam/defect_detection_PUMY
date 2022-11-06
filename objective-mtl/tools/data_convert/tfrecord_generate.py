import os
import argparse

from mtl.utils.gen_tfrecords.gen_tfrecord import generate_det_tfrecord
from mtl.utils.gen_tfrecords.gen_tfrecord import generate_cls_tfrecord
from mtl.utils.gen_tfrecords.gen_tfrecord import generate_seg_tfrecord
from mtl.utils.gen_tfrecords.gen_tfrecord import generate_mtl_tfrecord


def test_gen_tfreocrd():
    parser = argparse.ArgumentParser(
        description="Generate tfrecord for the mosaic detection dataset."
    )

    parser.add_argument(
        "--dataset_path",
        default="data/objdet-datasets/VOC",
        help="The root path of the given dataset",
    )
    parser.add_argument("--dataset_type", default="det", help="The type of the dataset")
    parser.add_argument(
        "--image_dir_name", default="images", help="The folder name of images."
    )
    parser.add_argument(
        "--label_dir_name", default="annotations", help="The folder name of labels."
    )
    parser.add_argument(
        "--split_dir_name",
        default="meta",
        help="The folder name for storing split lists.",
    )
    parser.add_argument(
        "--record_path",
        default="tfrecords",
        help="The recorded path of the given dataset",
    )
    parser.add_argument(
        "--is_file_ext",
        type=bool,
        default=False,
        help="Whether using extend file names",
    )
    parser.add_argument(
        "--label_format",
        default="normal",
        help="The annotation format for generating tfrecords, such as yolo, voc",
    )
    parser.add_argument("--split_str", default=" ", help="The split string")

    args = parser.parse_args()
    for list_file_name in os.listdir(
        os.path.join(args.dataset_path, args.split_dir_name)
    ):
        if list_file_name[-4:] == ".txt":
            if args.dataset_type == "det":
                generate_det_tfrecord(
                    args.dataset_path,
                    args.image_dir_name,
                    args.label_dir_name,
                    args.split_dir_name,
                    list_file_name,
                    args.record_path,
                    args.label_format,
                    args.split_str,
                )
            elif args.dataset_type == "cls":
                generate_cls_tfrecord(
                    args.dataset_path,
                    args.image_dir_name,
                    args.split_dir_name,
                    list_file_name,
                    args.record_path,
                    args.is_file_ext,
                )
            elif args.dataset_type == "seg":
                generate_seg_tfrecord(
                    args.dataset_path,
                    args.image_dir_name,
                    args.label_dir_name,
                    args.split_dir_name,
                    list_file_name,
                    args.record_path,
                )
            elif args.dataset_type == "mtl":
                generate_mtl_tfrecord(
                    args.dataset_path,
                    args.image_dir_name,
                    args.label_dir_name,
                    args.split_dir_name,
                    list_file_name,
                    args.record_path,
                )


if __name__ == "__main__":
    test_gen_tfreocrd()
