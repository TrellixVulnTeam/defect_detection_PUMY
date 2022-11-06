from pathlib import Path
import io
import os
import numpy as np
import cv2
import hashlib
import urllib
import shutil
import gzip
import tarfile
import zipfile
import inspect
from abc import ABCMeta, abstractmethod
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
from lxml.etree import Element, ElementTree, SubElement
from PIL import Image
from PIL import ImageFile
from torch.multiprocessing import Pool

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

from .handlers import JsonHandler, PickleHandler, YamlHandler
from .misc_util import is_str
from .path_util import check_file_exist, mkdir_or_exist
from .data_util import get_classes

ImageFile.LOAD_TRUNCATED_IMAGES = True

file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}

if TurboJPEG is not None:
    jpeg = TurboJPEG()
else:
    jpeg = None

supported_backends = ["cv2", "turbojpeg", "pillow"]

imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
}

imread_backend = "cv2"


def _jpegflag(flag="color", channel_order="bgr"):
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "color":
        if channel_order == "bgr":
            return TJPF_BGR
        elif channel_order == "rgb":
            return TJCS_RGB
    elif flag == "grayscale":
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img, flag="color", channel_order="bgr"):
    """Convert a pillow image to numpy array.
    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.
    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != "RGB":
            if img.mode != "LA":
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert("RGB")
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert("RGBA")
                img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag == "color":
            array = np.array(img)
            if channel_order != "rgb":
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag == "grayscale":
            img = img.convert("L")
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale" or "unchanged", ' f"but got {flag}"
            )
    return array


def imread(img_or_path, flag="color", channel_order="bgr", backend=None):
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
            Note that the `turbojpeg` backened does not support `unchanged`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `None`. If backend is None, the
            global imread_backend specified by ``use_backend()`` will be
            used. Default: None.

    Returns:
        ndarray: Loaded image array.
    """
    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f"backend: {backend} is not supported. Supported "
            "backends are 'cv2', 'turbojpeg', 'pillow'"
        )
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        check_file_exist(img_or_path, f"img file does not exist: {img_or_path}")
        if backend == "turbojpeg":
            with open(img_or_path, "rb") as in_file:
                img = jpeg.decode(in_file.read(), _jpegflag(flag, channel_order))
                if img.shape[-1] == 1:
                    img = img[:, :, 0]
            return img
        elif backend == "pillow":
            img = Image.open(img_or_path)
            img = _pillow2array(img, flag, channel_order)
            return img
        else:
            flag = imread_flags[flag] if is_str(flag) else flag
            img = cv2.imread(img_or_path, flag)
            if flag == IMREAD_COLOR and channel_order == "rgb":
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img
    else:
        raise TypeError(
            '"img" must be a numpy array or a str or ' "a pathlib.Path object"
        )


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.
    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        mkdir_or_exist(dir_name)
    return cv2.imwrite(file_path, img, params)


def imfrombytes(content, flag="color", channel_order="bgr", backend=None):
    """Read an image from bytes.
    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `None`. If backend is None, the
            global imread_backend specified by ``use_backend()`` will be
            used. Default: None.
    Returns:
        ndarray: Loaded image array.
    """
    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f"backend: {backend} is not supported. Supported "
            "backends are 'cv2', 'turbojpeg', 'pillow'"
        )
    if backend == "turbojpeg":
        img = jpeg.decode(content, _jpegflag(flag, channel_order))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
    elif backend == "pillow":
        buff = io.BytesIO(content)
        img = Image.open(buff)
        img = _pillow2array(img, flag, channel_order)
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == "rgb":
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    return img


def file_load(file, file_format=None, **kwargs):
    """Load data from json/yaml/pickle files.
    This method provides a unified api for loading data from serialized files.
    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if isinstance(file, str):
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def obj_dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.
    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.
    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dump to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def list_from_file(filename, prefix="", offset=0, max_num=0):
    """Load a text file and parse the content as a list of strings.
    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the begining of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
    Returns:
        list[str]: A list of strings.
    """
    print("filename: ", filename)
    cnt = 0
    item_list = []
    with open(filename, "r") as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip("\n"))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str):
    """Load a text file and parse the content as a dict.
    Each line of the text file will be two or more columns splited by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.
    Args:
        filename(str): Filename.
        key_type(type): Type of the dict's keys. str is user by default and
            type conversion will be performed if specified.
    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    with open(filename, "r") as f:
        for line in f:
            items = line.rstrip("\n").split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping


def rm_suffix(s, suffix=None):
    if suffix is None:
        return s[: s.rfind(".")]
    else:
        return s[: s.rfind(suffix)]


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url_to_file(url, fpath):
    with urllib.request.urlopen(url) as resp, open(fpath, "wb") as of:
        shutil.copyfileobj(resp, of)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from.
        root (str): Directory to place downloaded file in.
        filename (str | None): Name to save the file under.
            If filename is None, use the basename of the URL.
        md5 (str | None): MD5 checksum of the download.
            If md5 is None, download without md5 check.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f"Using downloaded and verified file: {fpath}")
    else:
        try:
            print(f"Downloading {url} to {fpath}")
            download_url_to_file(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    f" Downloading {url} to {fpath}"
                )
                download_url_to_file(url, fpath)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path, os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError(f"Extraction of {from_path} not supported")

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url,
    download_root,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished=False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    extract_archive(archive, extract_root, remove_finished)


def write_det_xml(out_file, width, height, bboxes, labels, dataset_name):
    node_root = Element("annotation")

    node_folder = SubElement(node_root, "folder")
    node_folder.text = "images"

    node_filename = SubElement(node_root, "filename")
    node_filename.text = os.path.basename(out_file)[:-4] + ".jpg"

    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, "width")
    node_width.text = str(width)
    node_height = SubElement(node_size, "height")
    node_height.text = str(height)
    class_names = get_classes(dataset_name)

    for i in range(len(bboxes)):
        node_object = SubElement(node_root, "object")

        node_name = SubElement(node_object, "name")
        node_name.text = class_names[labels[i]]

        node_truncated = SubElement(node_object, "truncated")
        node_truncated.text = "0"

        node_difficult = SubElement(node_object, "difficult")
        node_difficult.text = "0"

        node_bndbox = SubElement(node_object, "bndbox")
        node_xmin = SubElement(node_bndbox, "xmin")
        node_xmin.text = str(bboxes[i][0])
        node_ymin = SubElement(node_bndbox, "ymin")
        node_ymin.text = str(bboxes[i][1])
        node_xmax = SubElement(node_bndbox, "xmax")
        node_xmax.text = str(bboxes[i][2])
        node_ymax = SubElement(node_bndbox, "ymax")
        node_ymax.text = str(bboxes[i][3])

    # xml_str = tostring(node_root, pretty_print=True)
    tree = ElementTree(node_root)
    tree.write(out_file, pretty_print=True, xml_declaration=False, encoding="utf-8")


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.
    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.
    Args:
        db_path (str): Lmdb database path.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.
    Attributes:
        db_path (str): Lmdb database path.
    """

    def __init__(self, db_path, readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError("Please install lmdb to enable LmdbBackend.")

        self.db_path = str(db_path)
        self._client = lmdb.open(
            self.db_path, readonly=readonly, lock=lock, readahead=readahead, **kwargs
        )

    def get(self, filepath):
        """Get values according to the filepath.
        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
        """
        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode("ascii"))
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, "r") as f:
            value_buf = f.read()
        return value_buf


class FileClient:
    """A general file client to access files in different backend.
    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.
    Attributes:
        backend (str): The storage backend type. Options are "disk", "ceph",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {"disk": HardDiskBackend, "lmdb": LmdbBackend}

    def __init__(self, backend="disk", **kwargs):
        if backend not in self._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones"
                f" are {list(self._backends.keys())}"
            )
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    @classmethod
    def _register_backend(cls, name, backend, force=False):
        if not isinstance(name, str):
            raise TypeError(
                "the backend name should be a string, " f"but got {type(name)}"
            )
        if not inspect.isclass(backend):
            raise TypeError(f"backend should be a class but got {type(backend)}")
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f"backend {backend} is not a subclass of BaseStorageBackend"
            )
        if not force and name in cls._backends:
            raise KeyError(
                f"{name} is already registered as a storage backend, "
                'add "force=True" if you want to override it'
            )

        cls._backends[name] = backend

    @classmethod
    def register_backend(cls, name, backend=None, force=False):
        """Register a backend to FileClient.
        This method can be used as a normal class method or a decorator.
        .. code-block:: python
            class NewBackend(BaseStorageBackend):
                def get(self, filepath):
                    return filepath
                def get_text(self, filepath):
                    return filepath
            FileClient.register_backend('new', NewBackend)
        or
        .. code-block:: python
            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):
                def get(self, filepath):
                    return filepath
                def get_text(self, filepath):
                    return filepath
        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
        """
        if backend is not None:
            cls._register_backend(name, backend, force=force)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force)
            return backend_cls

        return _register

    def get(self, filepath):
        return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return ".zip@" in img_or_path


class ZipReader(object):
    """A class to read zipped files"""

    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, "r")
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index("@")
        assert pos_at != -1, (
            "character '@' is not found from the given path '%s'" % path
        )

        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1 :]
        folder_path = str.strip(folder_path, "/")
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, "/")
            if (
                file_foler_name.startswith(folder_path)
                and len(os.path.splitext(file_foler_name)[-1]) == 0
                and file_foler_name != folder_path
            ):
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1 :])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        if extension is None:
            extension = [".*"]
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, "/")
            if (
                file_foler_name.startswith(folder_path)
                and str.lower(os.path.splitext(file_foler_name)[-1]) in extension
            ):
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1 :])

        return file_lists

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

    @staticmethod
    def imread(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        try:
            im = Image.open(io.BytesIO(data))
        except:
            print("ERROR IMG LOADED: ", path_img)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im


class CacheFiles(object):
    def __init__(self, cache_dir="/dockerdata/CacheOriginDir", num_proc=5):
        super(CacheFiles, self).__init__()

        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.cache_dict = dict()

    def _check_file(self, file_path):
        file_name = os.path.basename(file_path)
        cache_file_path = os.path.join(self.cache_dir, file_name)
        if os.path.exists(cache_file_path):
            if os.path.getsize(cache_file_path) == os.path.getsize(file_path):
                self.cache_dict[file_path] = cache_file_path

    def cache(self, file_list):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            p = Pool(self.num_proc)
            for file_path in file_list:
                cache_file_path = os.path.join(
                    self.cache_dir, os.path.basename(file_path)
                )
                if not os.path.exists(cache_file_path) or os.path.getsize(
                    cache_file_path
                ) != os.path.getsize(file_path):
                    p.apply_async(os.system, (f"cp {file_path} {self.cache_dir}/",))
            p.close()

    def get(self, file_path):
        if file_path in self.cache_dict:
            return self.cache_dict[file_path]
        else:
            self._check_file(file_path)
            return file_path
