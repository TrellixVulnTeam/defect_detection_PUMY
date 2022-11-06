import argparse

from mtl.utils.gen_tfrecords.gen_tfrecord import generate_lupinus_tfrecord


def gen_lupinus_dataset():
    """Main process for spliting the datasets"""

    parser = argparse.ArgumentParser(
        description="Split the detection dataset into train and val sets."
    )

    parser.add_argument(
        "--dataset-path", default="", help="The path of the given dataset."
    )
    parser.add_argument("--data-dir", help="The folder name of images and annotations.")
    parser.add_argument(
        "--record-dir",
        default="tfrecords",
        help="The folder name for storing tfrecords.",
    )
    parser.add_argument(
        "--class-names-file",
        default="class_names_list.txt",
        help="The file name for storing class names list.",
    )
    parser.add_argument(
        "--store-file", action="store_true", help="Whether store file for class-names"
    )

    args = parser.parse_args()

    generate_lupinus_tfrecord(
        args.dataset_path,
        args.data_dir,
        args.record_dir,
        args.class_names_file,
        args.store_file,
    )


if __name__ == "__main__":
    gen_lupinus_dataset()
