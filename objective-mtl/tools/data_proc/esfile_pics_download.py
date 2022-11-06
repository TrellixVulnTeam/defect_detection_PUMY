import os
import traceback
import xlrd

from mtl.utils.parallel_util import get_pic_by_url_list


def download_imgs_from_esfile(es_file_path, sheet_name, download_path):
    book = xlrd.open_workbook(es_file_path)
    sheet = book.sheet_by_name(sheet_name)

    name_list = []
    url_list = []
    for i in range(1, sheet.nrows):
        try:
            down_file_name = sheet.row_values(i)[0].strip()
            down_url = sheet.row_values(i)[1].strip()
            if down_file_name != "" and down_url != "":
                name_list.append(down_file_name + ".jpg")
                url_list.append(down_url)

        except OSError:
            print("---------------------------------------------------------")
            print("Exeception information:")
            print(traceback.format_exc())
            print("Related Path: " + down_file_name)
            print("---------------------------------------------------------")

    if not os.path.exists(download_path):
        print("Download folder not exist, try to create it.")
        os.makedirs(download_path)

    get_pic_by_url_list(download_path, name_list, url_list)


if __name__ == "__main__":
    ESFILE_PATH = "data/objcls-datasets/xxx.xlsx"
    TABLE_NAME = "Sheet1"
    IMG_SAVE_PATH = "data/objcls-datasets/xxx/url_images"
    download_imgs_from_esfile(ESFILE_PATH, TABLE_NAME, IMG_SAVE_PATH)
