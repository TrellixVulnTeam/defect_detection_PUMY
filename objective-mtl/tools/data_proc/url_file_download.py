import os
import xlrd
import traceback
import urllib.request
import ssl


def get_pic_by_url(download_save_path, name_list, url_list):
    if not os.path.exists(download_save_path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(download_save_path)

    for j, url_path in enumerate(url_list):

        print("Try downloading file: {}".format(url_path))
        file_path = download_save_path + "/" + name_list[j]
        if os.path.exists(file_path):
            print("File have already exist. skip")
        else:
            try:
                # 不进行验证，直接下载
                ssl._create_default_https_context = ssl._create_unverified_context

                urllib.request.urlretrieve(url_path, filename=file_path)
            except Exception:
                print("---------------------------------------------------------")
                print("Exeception information:")
                print(traceback.format_exc())
                print("URL Path: " + url_path)
                print("---------------------------------------------------------")


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

        except Exception:
            print("---------------------------------------------------------")
            print("Exeception information:")
            print(traceback.format_exc())
            print("Related Path: " + down_file_name)
            print("---------------------------------------------------------")

    get_pic_by_url(download_path, name_list, url_list)


def download_imgs_from_txtfile(txt_file_path, download_path, anno_save_path):
    img_anno_info = open(txt_file_path, "r").readlines()

    name_list = []
    url_list = []
    info_list = []
    for img_anno in img_anno_info:
        try:
            img_info = img_anno.split("\t")
            if len(img_info) > 1:
                down_url = img_info[0].strip()
                down_file_name = down_url.split("/")[-1]
                if down_file_name != "" and down_url != "":
                    name_list.append(down_file_name)
                    url_list.append(down_url)

                    img_info[0] = "oid_val/" + down_file_name
                    info_list.append(img_info)

        except Exception:
            print("---------------------------------------------------------")
            print("Exeception information:")
            print(traceback.format_exc())
            print("Related URL Path: " + down_url)
            print("---------------------------------------------------------")

    # get_pic_by_url(download_path, name_list, url_list)

    with open(anno_save_path, "w") as fw:
        for anno_info in info_list:
            if not os.path.isfile(
                os.path.join(download_path, anno_info[0].split("/")[-1])
            ):
                continue

            for i, anno_element in enumerate(anno_info):
                fw.write(anno_element)
                if i < len(anno_info) - 1:
                    fw.write("\t")


if __name__ == "__main__":

    es_file = "data/objcls-datasets/xxxx/xxxx.xlsx"
    table_name = "Sheet1"
    folder_path = "data/objcls-datasets/xxxx/xxxx"
    download_imgs_from_esfile(es_file, table_name, folder_path)

    txt_file = "data/objcls-datasets/xxxx/xxxx.txt"
    img_save_path = "data/objcls-datasets/xxxx/xxxx"
    anno_s_path = "data/objcls-datasets/xxxx/xxxx.txt"
    download_imgs_from_txtfile(txt_file, img_save_path, anno_s_path)
