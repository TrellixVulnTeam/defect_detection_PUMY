IMG_SUFFIX_LIST = [".jpg", ".jpeg", ".png", ".bmp"]


def gen_labelme_json_model(img_size, img_path) -> dict:
    instance = {'version': '1.0',
                'shapes': [],
                'imageData': None,
                'imageWidth': img_size.get('width') if isinstance(img_size, dict) else None,
                'imageHeight': img_size.get('height') if isinstance(img_size, dict) else None,
                'imageDepth': img_size.get('depth') if isinstance(img_size, dict) else None,
                'imagePath': img_path}
    return instance
