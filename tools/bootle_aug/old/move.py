import shutil
fo1 = open("new/ImageSets/test.txt",'r')
lines2 = fo1.readlines()
for file in lines2:
	shutil.copy("new/annotations/" + file.replace("\n","") +".xml", "new/annotations/test")
print(lines2)