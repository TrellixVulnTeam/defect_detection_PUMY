import argparse
import cv2
import os

from mtl.utils.siliency_util import ImgNormalize, get_saliency_map


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True, help="path to input image")
    ap.add_argument("--test-type", default="itti", help="path to input image")
    args = ap.parse_args()
    img_normalizer = ImgNormalize()

    for img_file_name in os.listdir(args.image_dir):
        if img_file_name.endswith("jpg") or img_file_name.endswith("png"):
            image_path = os.path.join(args.image_dir, img_file_name)
            image = cv2.imread(image_path)
            sd_map = get_saliency_map(image, args.test_type, img_normalizer)

            # show the images
            cv2.imshow("IMG", image)
            cv2.imshow("SD_MAP", sd_map)
            cv2.waitKey(0)
