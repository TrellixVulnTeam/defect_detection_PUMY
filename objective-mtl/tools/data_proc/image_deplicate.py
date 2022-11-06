import os
import traceback
import numpy as np
from PIL import Image


def img_deplicate(img_dir):
    """Find the pair names with the same content in two folders"""

    img_features_list = []
    count = 0
    for img_file in os.listdir(img_dir):
        if img_file[-3:] != "jpg" and img_file[-3:] != "png":
            continue

        img_path = os.path.join(img_dir, img_file)

        # img = imread(img_path)
        try:
            pil_img = Image.open(img_path).convert("RGB")
            np_img = np.array(pil_img).astype(np.uint8)

            img_shape = np_img.shape
            # print(img_shape)
            img_sum_all = np.sum(np_img)
            img_sum_rgb = np.sum(np.sum(np_img, axis=1), axis=0)
            img_features_list.append(
                {
                    "name": img_file,
                    "feature": [
                        img_shape[0],
                        img_shape[1],
                        img_sum_all,
                        img_sum_rgb[0],
                        img_sum_rgb[1],
                        img_sum_rgb[2],
                    ],
                }
            )
            count += 1
            if count % 100 == 0:
                print("Processed %s images" % count)
        except Exception:
            print("---------------------------------------------------------")
            print("Exeception information:")
            print(traceback.format_exc())
            print("Related Path:", img_path)
            print("---------------------------------------------------------")

    for img_file in os.listdir(img_dir):
        if img_file[-3:] != "jpg" and img_file[-3:] != "png":
            continue
        img_path = os.path.join(img_dir, img_file)
        if not os.path.isfile(img_path):
            continue

        try:
            # img = imread(img_path)
            pil_img = Image.open(img_path).convert("RGB")
            np_img = np.array(pil_img).astype(np.int)

            img_shape = np_img.shape
            # print(img_shape)
            img_sum_all = np.sum(np_img)
            img_sum_rgb = np.sum(np.sum(np_img, axis=1), axis=0)

            img_feature_2 = [
                img_shape[0],
                img_shape[1],
                img_sum_all,
                img_sum_rgb[0],
                img_sum_rgb[1],
                img_sum_rgb[2],
            ]
            area_2 = img_shape[0] * img_shape[1]

            for i, img_feature in enumerate(img_features_list):
                if img_feature["name"] == img_file:
                    continue
                matched = True

                area = img_feature["feature"][0] * img_feature["feature"][1]

                for j in range(3, 6):
                    # print(abs(img_feature_2[j] - img_feature["feature"][j]))
                    if (
                        abs(
                            img_feature_2[j] / area_2 - img_feature["feature"][j] / area
                        )
                        > 2
                    ):
                        matched = False
                        break
                if matched:
                    print(img_feature["name"])
                    os.remove(os.path.join(img_dir, img_feature["name"]))
                    img_features_list.pop(i)

        except Exception:
            print("---------------------------------------------------------")
            print("Exeception information:")
            print(traceback.format_exc())
            print("Related Path:", img_path)
            print("---------------------------------------------------------")


if __name__ == "__main__":
    # '''Match the names according to image contents'''
    root_path = "data/objcls-datasets/xxxx/xxxx"
    img_deplicate(root_path)
