import os
import sys
import random
from PIL import Image
import csv

training_data = []
test_data = []
validation_data = []
test_y = []
training_y = []
FILEDIR='trainResized/'

for f in [f for f in os.listdir(FILEDIR) if not (f.startswith('.'))]:
    with Image.open(FILEDIR+f).convert('L') as im:
    	with open('trainLabels.csv','r') as x:
    		for i in x:
    			s=i.rstrip().split(',')
    			if f[0:-4]==s[0]:
    				val=str(s[1])
	        pix_array=list(sum([[int(f[0:-4])],[val],list(im.getdata())],[]))
        im.show()
        break
    if(random.random() > 0.2):
        training_data.append(pix_array)
    else:
        test_data.append(pix_array)
# x=["File","Value"]
# for i in range(1,401):
#     x.append(str(i))
# with open('training_data.csv','w') as f:
#     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#     wr.writerow(x)
#     for i in training_data:
#         wr.writerow(i)
# with open('test_data.csv','w') as f:
#     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#     wr.writerow(x)
#     for i in test_data:
#         wr.writerow(i)
