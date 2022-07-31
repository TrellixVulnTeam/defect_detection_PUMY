import os
path = "/home/zhang/datasets/tile/annotations/train/"
train = 538
test = 538
val = 4312

train = open("/home/zhang/datasets/tile/train.txt", mode="w+", encoding="utf-8")
for f in os.listdir(path):
    line_data = train.readlines()
    f = "/home/zhang/datasets/tile/image/" + f
    if len(line_data) != 0:
        train.write(f)
    else:
        train.write(f)


train.close()







