import os
import json
import xmltodict

from mtl.utils.io_util import file_load


# convert xml to json
def xml_to_json(xml_str):
    # xml parser
    xml_parse = xmltodict.parse(xml_str)
    # dumps() convert dict to json format, loads() convert json to dict format
    json_str = json.dumps(xml_parse, indent=1)
    return json_str


# convert json to xml
def json_to_xml(json_str):
    # using unparse() in xmltodict to convert json to xml
    xml_str = xmltodict.unparse(json_str, pretty=1)
    return xml_str


def remove_key_from_json(json_path):
    for sub_dir in os.listdir(json_path):
        if sub_dir.startswith("."):
            continue
        sub_path = os.path.join(json_path, sub_dir)
        for json_file in os.listdir(sub_path):
            if json_file.endswith(".json"):
                json_file_path = os.path.join(sub_path, json_file)
                anno_data = file_load(json_file_path)
                if "imageData" in anno_data:
                    anno_data.pop("imageData")
                with open(json_file_path, "w") as fj:
                    json.dump(anno_data, fj)


def test_main():

    b = """<?xml version="1.0" encoding="utf-8"?>
            <user_info>
                <id>12</id>
                <name>Tom</name>
                <age>12</age>
                <height>160</height>
                <score>100</score>
                <variance>12</variance>
            </user_info>
        """

    a = {
        "user_info": {
            "file_name": 12,
            "name": "Tom",
            "age": 12,
            "height": 160,
            "score": 100,
            "variance": 12,
        }
    }

    print("---------------------------split----------------------------------")
    print(xml_to_json(b))
    print("---------------------------split----------------------------------")
    print(json_to_xml(a))
    print("---------------------------split----------------------------------")


if __name__ == "__main__":
    # test_main()
    json_dir = "xxx"
    remove_key_from_json(json_dir)
