class_list=(
    "baidian" \
    "zangwu" \
    "huashang" \
    "aotu" \
    "cashang" \
    "heidian" \
    "yise" \
    "bengbian" \
    "yahen" \
    "juchi" \
    "fameng" \
#    "dingyashang" \
#    "dingshuiyin" \
#    "cehuashang" \
#    "dingyise" \
#    "dingliangyin" \
#    "queliao" \
#    "mhc" \
#    "liuwen" \
    )
image_width=768
image_high=768
project_name="Apple"
folder_name="DM"
# shellcheck disable=SC2068
for class_name in ${class_list[@]}
do
  echo $class_name
  echo /home/zhang/Dataset/Other/${class_name}/${project_name}/${folder_name}/
  echo /home/zhang/Project/${project_name}/${folder_name}
  echo /home/zhang/Dataset/Other/${class_name}/${project_name}/${folder_name}/${image_width}X${image_high}/
  python preprocess.py \
  --work_dir /home/zhang/Project/${project_name}/${folder_name} \
  --output_dir /home/zhang/Dataset/Other/${class_name}/${project_name}/${folder_name}/${image_width}X${image_high}/ \
  --labels_filter $class_name \
  --image_width ${image_width} \
  --image_height ${image_high}
done
