import os
import argparse
import numpy as np


def create_filename_lists(
    dataset_path, img_dir_name, label_dir_name, val_percentage=10, test_percentage=0
):
    """Create data lists with all dictionaries in the data path.

    :param dataset_path: the path of dataset
    :param img_dir_name: the folder name for images
    :param label_dir_name: the folder name for labels
    :param val_percentage: the percentage of validation data
    :param test_percentage: the percentage of testing data
    :return: the dict of train, val and test lists of file names
    """
    result = {}
    train_file_list = []
    val_file_list = []
    test_file_list = []
    img_path = os.path.join(dataset_path, img_dir_name)
    label_path = os.path.join(dataset_path, label_dir_name)

    for file_name in os.listdir(img_path):
        # achieve the label files
        if not os.path.isfile(os.path.join(label_path, file_name[:-4] + ".txt")):
            continue

        # only store the file names
        chance = np.random.randint(100)
        if chance < 100 - val_percentage - test_percentage:
            train_file_list.append(file_name[:-4])
        elif chance < 100 - test_percentage:
            val_file_list.append(file_name[:-4])
        else:
            test_file_list.append(file_name[:-4])

    result["train"] = train_file_list
    result["val"] = val_file_list
    result["test"] = test_file_list

    return result


def split_dataset():
    """Main process for spliting the datasets"""

    parser = argparse.ArgumentParser(
        description="Split the detection dataset into train and val sets."
    )

    parser.add_argument(
        "--dataset_path",
        default="data/objdet-datasets/VOC/VOCdevkit/",
        help="The path of the given dataset.",
    )
    parser.add_argument(
        "--image_dir_name", default="Images", help="The folder name of images."
    )
    parser.add_argument(
        "--label_dir_name", default="Annotations", help="The folder name of labels."
    )
    parser.add_argument(
        "--store_dir_name",
        default="ImageSets",
        help="The folder name for storing splitted files.",
    )
    parser.add_argument(
        "--val_percentage", default=10, type=int, help="The percentage for validation."
    )
    parser.add_argument(
        "--test_percentage", default=0, type=int, help="The percentage for testing."
    )
    parser.add_argument(
        "--is_new",
        default=True,
        type=bool,
        help="Create the split files or extend them?",
    )

    args = parser.parse_args()

    if args.is_new:
        write_flag = "w"
    else:
        write_flag = "a"

    split_result = create_filename_lists(
        args.dataset_path,
        args.image_dir_name,
        args.label_dir_name,
        args.val_percentage,
        args.test_percentage,
    )

    train_file_path = os.path.join(args.dataset_path, args.store_dir_name, "train.txt")
    f_train = open(train_file_path, write_flag)
    for item in split_result["train"]:
        f_train.write(item + "\n")
    f_train.close()

    val_file_path = os.path.join(args.dataset_path, args.store_dir_name, "val.txt")
    f_val = open(val_file_path, write_flag)
    for item in split_result["val"]:
        f_val.write(item + "\n")
    f_val.close()

    test_file_path = os.path.join(args.dataset_path, args.store_dir_name, "test.txt")
    f_test = open(test_file_path, write_flag)
    for item in split_result["test"]:
        f_test.write(item + "\n")
    f_test.close()


if __name__ == "__main__":
    split_dataset()
