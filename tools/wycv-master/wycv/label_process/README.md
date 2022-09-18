# Label Process 文档
- **Tips:** 所有脚本参数均可使用`python ***.py --help`查阅。
## Prerequisite
- `git clone https://gitlab.micro-i.com.cn:9443/wyi/wycv.git`
- `pip install -r wycv/requirements.txt`
- `cd wycv`
- `python setup.py install`

## gen_json/gen_json.py

- **描述**
  - 用于 *Tianchi* 竞赛的`.csv`格式标签转换为`labelme`格式。
  - 用法
    - **参数解释**
    > `-w / --work_dir` 用于指定数据源路径，即图像所在文件夹，该参数为必须项；
    >
    > `-o / --output_path` 用于指定输出路径，当指定`output_path`已存在时，输出将附加进该文件夹，如有重名将覆盖，该参数缺省为`work_dir`同级目录下的`labelme`文件夹；
    >
    > `-d / --dry_run` 使用该参数时，在执行脚本前将先清除`output_path`（**请慎用该参数**）；
    >
    > `--csv_file` 用于指定`work_dir`中图片所对应的`.csv`标注文件；
    - **使用示例: 将 *赛道2* 图像转`labelme`格式**
    > run `python gen_json_for_tianchi.py -w /home/adt/Downloads/Tianchi/2_images/ --csv_file /home/adt/Downloads/Tianchi/2train_rname.csv`

## gen_json/gen_json.py

- **描述**
  - 用于将王老师团队或MEB团队标注产生的`.xml`文件，数据中心导出的`.xls/.xlsx`文件以及labelme标注格式的`.json`文件转换为统一标准的labelme格式的`.json`标注文件（输出文件结构如下）。
  - 用于 *Tianchi* 竞赛的`.csv`格式标签转换为`labelme`格式。
    ```shell script
    - output_path
      - labelme  # 输出图片数据与 labelme 格式 json 数据
        - ****-**-*.jpg
        - ****-**-*.jpg
        - ****-**-*.json
        - ****-**-*.json
        - ...
    ```

- **用法**
  - **参数解释**
    > `-w / --work_dir` 用于指定数据源路径，具体用法可见下方场景示例，该参数为必须项；
    > 
    > `-o / --output_path` 用于指定输出路径，当指定`output_path`已存在时，输出将附加进该文件夹，如有重名将覆盖，该参数缺省为`work_dir`同级目录下的`labelme`文件夹；
    > 
    > `-r / --recursive` 使用该参数时，`gen_json.py`脚本扫描目标将为`work_dir`所包含的文件夹（适用于转换多批数据的情况）；
    > 
    > `-d / --dirty_run` 使用该参数时，在执行脚本前将先清除`output_path`（**请慎用该参数**）；
    > 
    > `-x / --excel_file` 在转换数据中心导出的`.xls/.xlsx`文件时，用于指定`work_dir`中图片所对应的`.xls/.xlsx`标注文件；
    >
    > `--csv_file` 用于指定`work_dir`中图片所对应的`.csv`标注文件；
    > 
    > `-c / --map_config` 在转换数据中心导出的`.xls/.xlsx`文件时，用于指定`excel`标注文件的映射关系文件，该文件须包含`label_dict`与`excel`文件中的`title`与实际含义的映射关系（示例可见`gen_json/map_config.yml`）；
    > 
    > `-p / --process_num` 用于指定进程数量，该参数缺省为`8`。

  - **使用示例: 将天池 *赛道2* 图像转`labelme`格式**
    > run `python gen_json.py -w /home/adt/Downloads/Tianchi/2_images/ --csv_file /home/adt/Downloads/Tianchi/2train_rname.csv`

  - **场景 1: 多批数据（如下格式）转换。在该场景下，仅转换王老师团队或MEB团队标注产生的`.xml`文件与labelme标注格式的`.json`文件**
    ```shell script
    - data_source
      - data20210511  # xml格式数据
        - outputs
          - ****-**-*.xml
          - ****-**-*.xml
          - ...
        - ****-**-*.jpg
        - ****-**-*.png
        - ...
      - source20210513   # labelme格式数据
        - a.jpg
        - a.json
        - b.bmp
        - b.json
        - ...
      - excel_format_data   # excel格式数据，多批数据转换时，该会被忽略
        - ****-**-**.jpg
        - ***.xlsx
    ```
    > run `python gen_json.py -w /data_source -o $OUTPUT_PATH -r -d -p 8`

  - **场景 2: 转换王老师团队或MEB团队标注产生的`.xml`或labelme标注格式的`.json`单批数据**
    > run `python gen_json.py -w /data_source/data20210511 -o $OUTPUT_PATH -p 8`

  - **场景 3: 转换单批数据中心导出的`.xls/.xlsx`文件数据（数据仅建议用于检查核对，不建议用于训练）**
    > run `python gen_json.py -w /data_source/excel_format_data -x $EXCEL_FILE -c $MAP_CONFIG_FILE -o $OUTPUT_PATH -p 8`

## label_converter/label_converter.py

- **描述**
  - 用于将`gen_json.py`得到的标准输出转换为`coco`格式（或`yolo_text`格式）的标注文件（输出文件结构如下），并自动划分训练集与验证集。
  ```shell script
    - output_path
      - coco
        - annotations  # 合成的大的coco文件
          - instances_train.json
          - instances_val.json
        - train
          - ***.jpg
          - ***.json
          - ...
        - val
          - ***.jpg
          - ***.json
          - ...
        - stuffthingmaps
          - train
            - mask    #彩图mask
            - labels  #灰度mask
          - val
            - mask    #彩图mask
            - labels  #灰度mask
          - except  #发生异常的图片
      - yolo
        - images
          - train
            - ****-**-*.jpg
            - ...
          - val
            - ****-**-*.jpg
            - ...
        - labels
          - train
            - ****-**-*.txt
            - ...
          - val
            - ****-**-*.txt
            - ...
  ```

- **用法**
  - **配置文件及参数解释**
    > 配置文件模版见`label_converter/convert_config.yml`
    ```yaml
    work_dir: "/home/adt/dataset/data/labelme"  # 包含图片和 labelme 格式标注的 json 文件的文件夹路径
    output_path: "/home/adt/dataset/output"  # 结果输出路径

    convert_params:
      target_format: 'coco'  # 转出格式，目前支持 'coco' 与 'yolo'
      remain_bg: True  # 仅转出格式为 'coco' 时有效，控制是否在标注文件的 'categories' 中增加 'background' 类，缺省为 True
      isRLE: False  # 仅转出格式为 'coco' 时有效，控制标注格式是否采用 'RLE' 格式，缺省为 False
      gen_mask: False  # 仅转出格式为 'coco' 时有效，控制是否生成实例对应 label mask (灰阶图)，缺省为True
      color_mask: False  # 仅转出格式为 'coco' 时有效，控制是否生成实例对应彩色 mask (便于观察)，缺省为True
      line_width: 5  # 仅转出格式为 'coco' 时有效，控制绘制 mask 时的线条宽度，缺省为 5
    
    split_params:
      method: 'random_split'  # 训练集验证集划分方法，目前支持 'random_split' 与 'filter_split'
      val_ratio: 0.2  # 划分时验证集所占比例，缺省为 0
      random_seed: 40  # 仅划分方法为 'random_split' 时有效，随机种子，同一数据集多次运行该脚本时使用相同随机种子与相同划分比例可使划分结果一致
      filter_label: [1, 2, 3, 4]  # 仅划分方法为 'filter_split' 时有效，过滤标签，用于划分时优先确保所设标签服从同一分布
      level: 3  # 仅划分方法为 'filter_split' 时有效，用于设定从前几个 level 的数据中取验证集，缺省为 3 (该值须大于等于1且小于等于3)
    
    label_dict:  # 完整的标签字典，适用于数据集标签不齐全的情况
      'yise': 1
      'baisezaodian': 2
      'shuiyin': 3
      'guashang': 4
      'cashang': 5
      'heidian': 6
      'shahenyin': 7
      'yiwu': 8
      'daowen': 9
      ...
    ```
  - 配置文件定义好之后通过如下命令执行（其中，`-p / --process_num`用于指定运行线程数量）
    > 
    > `python label_converter.py -c /home/adt/dataset/convert_config.yml -p 8`
