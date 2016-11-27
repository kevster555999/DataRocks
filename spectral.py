import csv
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def laplacian(np_arr):
	x = np_arr.reshape(20,20)
	for i in np_arr:
		break
	pass
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
test=[[0,0,0],[0,23,0],[0,23,0]]
test = np.asarray(test).reshape(3,3)
print test


graph=image.img_to_graph(np_x[1])
print graph
#sc = SpectralClustering(n_clusters=2,assign_lables='kmeans',eigen_solver='arpack')
#sc.fit_predict(graph)

#print sc.labels_
labels = spectral_clustering(graph,n_clusters=2,eigen_solver='arpack')

print labels.reshape(20,20)
plt.matshow(labels.reshape(20,20))
plt.matshow(np_x[1].reshape(20,20))
plt.show()
print chr(y[1])
#print len(labels)
#print labels

#mask = np_x[1].astype(bool)
#print mask.shape
#label_im = -np.ones(mask.shape)
#label_im[mask] = labels
#plt.matshow(label_im)