import argparse

from mtl.utils.gen_tfrecords.gen_tfrecord import generate_coco_tfrecord


def gen_coco_dataset():
    """Main process for spliting the datasets"""

    parser = argparse.ArgumentParser(
        description="Split the detection dataset into train and val sets."
    )

    parser.add_argument(
        "--dataset_path",
        default="data/objdet-datasets/COCO",
        help="The path of the given dataset.",
    )
    parser.add_argument(
        "--train_img_dir", default="train2017", help="The folder name of images."
    )
    parser.add_argument(
        "--train_anno_path",
        default="annotations/instances_train2017.json",
        help="The folder name of images.",
    )
    parser.add_argument(
        "--val_img_dir", default="val2017", help="The folder name of labels."
    )
    parser.add_argument(
        "--val_anno_path",
        default="annotations/instances_val2017.json",
        help="The folder name of images.",
    )
    parser.add_argument(
        "--record_path",
        default="tfrecords",
        help="The folder name for storing tfrecords.",
    )

    args = parser.parse_args()

    generate_coco_tfrecord(
        args.dataset_path,
        args.train_img_dir,
        args.train_anno_path,
        args.record_path,
        "train",
    )

    generate_coco_tfrecord(
        args.dataset_path, args.val_img_dir, args.val_anno_path, args.record_path, "val"
    )


if __name__ == "__main__":
    gen_coco_dataset()
