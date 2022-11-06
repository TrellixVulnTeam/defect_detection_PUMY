# VOC数据集格式
# VOC/VOCdevkit
# |--VOC2007
#    |--Annotations/Imagesets/JPEGImages
# |--VOC2012
#    |--Annotations/Imagesets/JPEGImages

DATASET_DIR='data/objdet-datasets/VOC/VOCdevkit/'
if [ ! -d $DATASET_DIR/tfrecords ];then
mkdir -p $DATASET_DIR/tfrecords
fi

SOURCE_DIR="data/objdet-datasets/VOC/VOCdevkit/VOC2007/ImageSets/Main"
SPLIT_DIR="data/objdet-datasets/VOC/VOCdevkit/VOC2007/ImageSets/TrainVal"
if [ ! -d $SPLIT_DIR ];then
    mkdir -p $SPLIT_DIR
fi
cp $SOURCE_DIR/train.txt $SPLIT_DIR/train07.txt
cp $SOURCE_DIR/val.txt $SPLIT_DIR/val07.txt
python3 tools/data_convert/tfrecord_generate.py \
--dataset_path  $DATASET_DIR \
--dataset_type 'det' --image_dir_name 'VOC2007/JPEGImages' \
--label_dir_name 'VOC2007/Annotations' \
--split_dir_name 'VOC2007/ImageSets/TrainVal' --record_path 'tfrecords' --label_format 'voc'

SOURCE_DIR2012="data/objdet-datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main"
SPLIT_DIR2012="data/objdet-datasets/VOC/VOCdevkit/VOC2012/ImageSets/TrainVal"
if [ ! -d $SPLIT_DIR2012 ];then
    mkdir -p $SPLIT_DIR2012
fi
cp $SOURCE_DIR2012/train.txt $SPLIT_DIR2012/train12.txt
cp $SOURCE_DIR2012/val.txt $SPLIT_DIR2012/val12.txt
python3 tools/data_convert/tfrecord_generate.py \
--dataset_path $DATASET_DIR \
--dataset_type 'det' --image_dir_name 'VOC2012/JPEGImages' \
--label_dir_name 'VOC2012/Annotations' \
--split_dir_name 'VOC2012/ImageSets/TrainVal' --record_path 'tfrecords' --label_format 'voc'
