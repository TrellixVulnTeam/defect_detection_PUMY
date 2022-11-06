import os
import logging
import struct
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from .gen_tfrecords.wy_example_pb2 import Example


def tfrecord2idx(tfrecord, idxfile):
    """
    refer :  https://github.com/NVIDIA/DALI/blob/master/tools/tfrecord2idx
    """
    try:
        # check idxfile exist and size large than 0
        st = os.stat(idxfile)
        if st.st_size > 0:
            return idxfile
    except:
        # no found or size is zero regenerate
        pass

    samples = 0
    with open(tfrecord, "rb") as f:
        with open(idxfile, "w") as idx:
            while True:
                current = f.tell()
                byte_len_crc = f.read(12)
                # eof
                if len(byte_len_crc) == 0:
                    break
                if len(byte_len_crc) != 12:
                    logging.error(
                        "read byte_len_crc failed, file:%s, num:%d pos:%s byte_len_crc:%s"
                        % (tfrecord, samples, f.tell(), len(byte_len_crc))
                    )
                    break
                proto_len = struct.unpack("L", byte_len_crc[:8])[0]
                buffer = f.read(proto_len + 4)
                if len(buffer) != proto_len + 4:
                    logging.error(
                        "read proto_len failed, file:%s, num:%d pos:%s proto_len:%s"
                        % (tfrecord, samples, f.tell(), proto_len)
                    )
                    break
                idx.write(str(current) + " " + str(f.tell() - current) + "\n")
                samples += 1
    if samples == 0:
        logging.error("no idx found,  file:%s" % tfrecord)
        os.remove(idxfile)
        return None
    logging.info("idx generate done, samples:%s file:%s" % (samples, idxfile))
    return idxfile


class TFRecordDataSet(Dataset):
    def __init__(self, tfrecords):
        tfindexs = [tfrecord2idx(f, f.replace(".tfrecord", ".idx")) for f in tfrecords]
        self.data_infos = []
        self.samples = 0
        for index, tffile in zip(tfindexs, tfrecords):
            idx = []
            with open(index) as idxf:
                for line in idxf:
                    offset, _ = line.split(" ")
                    idx.append(offset)
            self.samples += len(idx)
            print("load %s, samples:%s" % (tffile, len(idx)))
            self.data_infos.append((idx, tffile))

    def __len__(self):
        return self.samples

    def parser(self, feature_list):
        raise NotImplementedError("Must Implement parser")

    def get_record(self, f, offset):
        f.seek(offset)

        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack("Q", byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) < proto_len:
            print(
                "read pb_data err,proto_len:%s pb_data len:%s"
                % (proto_len, len(pb_data))
            )
            return None

        example = Example()
        example.ParseFromString(pb_data)
        # keep key value in order
        feature = sorted(example.features.feature.items())

        record = self.parser(feature)
        # print(record)
        return tuple(record)

    def __getitem__(self, index):
        for idx, tffile in self.data_infos:
            if index >= len(idx):
                index -= len(idx)
                continue
            f = open(tffile, "rb")

            offset = int(idx[index])
            return self.get_record(f, offset)

        print("bad index,", index)


class ImageTFRecordDataSet(TFRecordDataSet):
    def __init__(self, tfrecords, transforms=None):
        super(ImageTFRecordDataSet, self).__init__(tfrecords)
        self.transforms = transforms

    def parser(self, feature_list):
        """
        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        """
        for key, feature in feature_list:

            # for image file col
            if key == "name":
                name = feature.bytes_list.value[0].decode("UTF-8", "strict")
            elif key == "image":
                image = feature.bytes_list.value[0]
                if self.transforms is not None:
                    image = Image.open(BytesIO(image))
                    image = image.convert("RGB")
                    image = self.transforms(image)
            elif key == "bbox/class":
                obj_cls = feature.int64_list.value
            elif key == "bbox/xmin":
                obj_xmin = feature.int64_list.value
            elif key == "bbox/ymin":
                obj_ymin = feature.int64_list.value
            elif key == "bbox/xmax":
                obj_xmax = feature.int64_list.value
            elif key == "bbox/ymax":
                obj_ymax = feature.int64_list.value

        return name, image, obj_cls, obj_xmin, obj_ymin, obj_xmax, obj_ymax
