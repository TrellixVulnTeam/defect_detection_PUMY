import os
path = "/home/zhang/datasets/PCBData/images/val/"
txt = "/home/zhang/datasets/PCBData/images/val/val.txt"
fp = open(txt,"w")
for f in os.listdir(path):
    fp.write("\n"+ path + f)



fp.close()