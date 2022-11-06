import argparse

from mtl.utils.gen_tfrecords.gen_tfrecord import generate_object365_tfrecord


def gen_object365_dataset():
    """Main process for spliting the datasets"""

    parser = argparse.ArgumentParser(
        description="Split the detection dataset into train and val sets."
    )

    parser.add_argument(
        "--dataset_path",
        default="data/objdet-datasets/Object365",
        help="The path of the given dataset.",
    )
    parser.add_argument(
        "--train_img_dir", default="train", help="The folder name of images."
    )
    parser.add_argument(
        "--train_anno_path",
        default="annotations/zhiyuan_objv2_train.json",
        help="The folder name of images.",
    )
    parser.add_argument(
        "--val_img_dir", default="val", help="The folder name of labels."
    )
    parser.add_argument(
        "--val_anno_path",
        default="annotations/zhiyuan_objv2_val.json",
        help="The folder name of images.",
    )
    parser.add_argument(
        "--record_path",
        default="tfrecords",
        help="The folder name for storing tfrecords.",
    )

    args = parser.parse_args()

    generate_object365_tfrecord(
        args.dataset_path,
        args.train_img_dir,
        args.train_anno_path,
        args.record_path,
        "train",
    )

    generate_object365_tfrecord(
        args.dataset_path, args.val_img_dir, args.val_anno_path, args.record_path, "val"
    )


if __name__ == "__main__":
    # split_dataset()
    gen_object365_dataset()
