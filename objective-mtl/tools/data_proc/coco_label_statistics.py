from pycocotools.coco import COCO


def get_coco_label_stat(data_dir, data_type):
    ann_file = "{}/annotations/instances_{}.json".format(data_dir, data_type)

    # initialize COCO api for instance annotations
    coco = COCO(ann_file)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat["name"] for cat in cats]
    print("number of categories: ", len(cat_nms))
    print("COCO categories: \n", cat_nms)

    # 统计各类的图片数量和标注框数量
    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=cat_name)  # 1~90
        imgId = coco.getImgIds(catIds=catId)  # 图片的id
        annId = coco.getAnnIds(catIds=catId)  # 标注框的id
        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))


if __name__ == "__main__":
    data_dir = "data/objdet-datasets/COCO"
    data_type = "train2017"
    get_coco_label_stat(data_dir, data_type)
