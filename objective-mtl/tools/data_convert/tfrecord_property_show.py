import struct

from mtl.utils.tfrecord_util import tfrecord2idx
from mtl.utils.gen_tfrecords.wy_example_pb2 import Example


def get_info(tfrecord_path):
    tfindex = tfrecord2idx(tfrecord_path, tfrecord_path.replace(".tfrecord", ".idx"))
    idx = []
    with open(tfindex) as idxf:
        for line in idxf:
            offset, _ = line.split(" ")
            idx.append(offset)
    print("load %s, samples:%s" % (tfrecord_path, len(idx)))

    f = open(tfrecord_path, "rb")
    offset = int(idx[0])
    f.seek(offset)
    # length,crc
    byte_len_crc = f.read(12)
    proto_len = struct.unpack("Q", byte_len_crc[:8])[0]
    # proto,crc
    pb_data = f.read(proto_len)
    if len(pb_data) < proto_len:
        print(
            "read pb_data err,proto_len:%s pb_data len:%s" % (proto_len, len(pb_data))
        )
        return None

    example = Example()
    example.ParseFromString(pb_data)
    # keep key value in order
    feature_list = sorted(example.features.feature.items())

    for key, feature in feature_list:
        print(key, type(feature))


if __name__ == "__main__":
    TF_PATH = "data/objdet-datasets/COCO/tfrecords/val.tfrecord"
    get_info(TF_PATH)
