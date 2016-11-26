import csv
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering

x=[]
y=[]
with open('training_data.csv','r') as csvfile:
	csvfile.readline()
	f = csv.reader(csvfile, delimiter=',')
	for line in f:
		y.append(ord(line[1]))
		tmp=[]
		for i in line[2:]:
			tmp.append(int(i))
		x.append(tmp)
	
np_x = np.asarray(x)
#print np_x[1].reshape(20,20)
labels = spectral_clustering(np_x[0].reshape(20,20),n_clusters=3,eigen_solver='arpack')
#print len(labels)
print labels

#mask = np_x[1].astype(bool)
#print mask.shape
#label_im = -np.ones(mask.shape)
#label_im[mask] = labels
#plt.matshow(label_im)