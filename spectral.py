import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle
import collections
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import ndimage
from collections import defaultdict
from sklearn.cluster import KMeans
#import skimage
#from skimage import data
#from skimage import filters


def filter(img):
	return filters.threshold_otsu(img.reshape(20,20))

def mask(a):
	a.reshape(20,20)
	for i in range(20):
		for j in range(20):
			pass

def find_min_dist(img):
	min_dist=1000000,'None'
	for key in avg_img.keys():
		d = compute_distance(avg_img[key],img)
		d_i= compute_distance(avg_img[key],inverse_img(img))
#		print min_dist[0],d,d_i,chr(key)
		if d < min_dist[0]:
			min_dist=d,chr(key),0
		elif d_i < min_dist[0]:
			min_dist=d_i,chr(key),1
		else:
			min_dist=min_dist
	return min_dist

def compute_distance(a,b):
	assert len(a) == len(b)
	res=0
	for i in range(len(a)):
		res= (a[i]-b[i])*(a[i]-b[i])
	return math.sqrt(res)

def simple_binarization(x):
	res=[]
	for i in x:
		res.append(255 if i>127 else 0)
	return res

def full_binarization(x):
	res=[]
	for i in x:
		res.append(simple_binarization(i))
	return np.asarray(res)

def average_images(img_array):
	for image in img_array:
		try:
			if(compute_distance(res_image),image) < 40:
				res_image+=image
			pass
		except:
			res_image=image
	res_image=res_image*1.0 / len(img_array)
	return res_image

def inverse_img(img):
	tmp=0
	for i in range(len(img)):
		if img[i]>tmp:
			tmp=img[i]
	for i in range(len(img)):
		img[i]= (-1.0)*img[i]+tmp
	return img

def label_mapping(label,y_val):
	od=collections.OrderedDict()
	output_set=set([])
	label_set=set([])
	for i in range(len(label)):
		label_set.add(i)
	for i in range(48,58):
		output_set.add(i)
	for i in range(65,91):
		output_set.add(i)
	for i in range(97,123):
		output_set.add(i)
	result_list=[]
	all_data = np.column_stack((y_val,label))
	tmp=defaultdict(list)
	for i in all_data:
		tmp[i[0]].append(i[1])
	for i in tmp.keys():
		tmp2={}
		for j in tmp[i]:
			try:
				tmp2[j]+=1
			except:
				tmp2[j]=1
	od=collections.OrderedDict(sorted(tmp.items(),key=lambda t: t[1]))
	for key,val in od.iteritems():
		l=0,
		for key in tmp2.keys():
			if(tmp2[key] > l[0]):
				l=tmp2[key],i,key
		for key in tmp2.keys():
			if l[2] in label_set:
				label_set.remove(l[2])
				result_list.append(l)
				break
			l=l[0],l[1],key
	sum_=0
	return result_list

#second attempt at relabeling, it is fine to assign multiple y_vals to labels
def l_mapping(label,y_val):
	res=[]
	all_data = np.column_stack((label,y_val))
	label_dict=defaultdict(list)
	for i in all_data:
		label_dict[i[0]].append(i[1])
	for i in label_dict.keys():
		y_val_dict={}
		for j in label_dict[i]:
			try:
				y_val_dict[j]+=1
			except:
				y_val_dict[j]=1
		high=0,'na'
		for j in y_val_dict.keys():
			if y_val_dict[j] > high[0]:
				high=y_val_dict[j],j
		res.append((high[1],i))
	print res
	return res

def label_dict(labels):
	res=defaultdict(list)
	for i in labels:
		res[i[1]].append(i[0])
	print res.keys()
	return res
			#prob = tmp[key]/ np.linalg.norm(tmp[key])
			#prob = np.sort(prob)
def label_adjust(labels):
	output_set=set([])
	label_set=set([])
	for i in range(len(labels)):
		label_set.add(i)
	for i in range(48,58):
		output_set.add(i)
	for i in range(65,91):
		output_set.add(i)
	for i in range(97,123):
		output_set.add(i)
	assert len(output_set)==len(label_set)
	for label in labels:
		if label[2] > 5:
			output_set.remove(label[0])
			try:
				label_set.remove(label[1])
			except:
				print "already removed"
	print len(output_set)
	print len(label_set)

def spectral_clustering(x,y):
	connectivity = kneighbors_graph(x, n_neighbors=10,mode='connectivity', include_self=False)
	connectivity = 0.75 * (connectivity + connectivity.T)
	#print connectivity
	spectral = SpectralClustering(n_clusters=62,eigen_solver='arpack',affinity="nearest_neighbors")
	spectral.fit_predict(connectivity)
	print spectral.labels_
	print spectral.affinity_matrix_.shape
	lb= l_mapping(spectral.labels_,y)
	lab_d=label_dict(lb)

	tmp=0
	for i in range(len(x)):
		try:
			if(y[i] == lab_d[i]):
				tmp+=1
			else:
#				print test_y[i],lab_d[max_d[1]][0]
				pass
		except:
#			print lab_d[max_d[1]],max_d[1]
			pass
	print tmp*1.0/len(test_x)

def check_dist(a,y):
	for i in avg_img.keys():
		print chr(y),chr(i),compute_distance(a,avg_img[i])

x=[]
y=[]
t_x=[]
t_y=[]
full=defaultdict(list)

#instantiating a normal array of all our attributes
with open('training_data.csv','r') as csvfile:
	csvfile.readline()
	f = csv.reader(csvfile, delimiter=',')
	for line in f:
		y.append(ord(line[1]))
		tmp=[]
		for i in line[2:]:
			tmp.append(int(i))
		x.append(tmp)


with open('test_data.csv','r') as csvfile:
	csvfile.readline()
	f = csv.reader(csvfile, delimiter=',')
	for line in f:
		t_y.append(ord(line[1]))
		tmp=[]
		for i in line[2:]:
			tmp.append(int(i))
		t_x.append(tmp)

test_x=np.asarray(t_x)
test_y=np.asarray(t_y)
np_x=np.asarray(x)
np_y=np.asarray(y)
#instantiating a dictionary with y as keywords to train models
with open('training_data.csv','r') as csvfile:
	csvfile.readline()
	f = csv.reader(csvfile, delimiter=',')
	for line in f:
		full[ord(line[1])].append(map(lambda x: int(x),line[2:]))
for key in full.keys():
	full[key]=np.asarray(full[key])
#instantiating avg_images array
avg_img={}
for key in full.keys():
	avg_img[key]=average_images(full[key])

check_dist(np_x[1000],np_y[1000])


#plt.matshow(np.asarray(avg_img[65]).reshape(20,20))
#plt.matshow(np.asarray(avg_img[119]).reshape(20,20))

guess_centers=[]
for key in sorted(avg_img.keys()):
	guess_centers.append(avg_img[key])
guess_centers=np.asarray(guess_centers)
#K-means clustering here (Things to do -> find a label mapping and predict with cluster centers)
kmeans = KMeans(n_clusters=62,max_iter=1000000,init=guess_centers)
kmeans.fit_predict(full_binarization(np_x))
#print kmeans.labels_
print kmeans.cluster_centers_.shape
#print kmeans.inertia_

# lb= label_mapping(kmeans.labels_,np_y)
# #label_adjust(lb)
# for i in lb:
# 	print i


lb= l_mapping(kmeans.labels_,np_y)
lab_d=label_dict(lb)
#print lab_d

tmp=0
for i in range(len(test_x)):
	max_d=100000000,0
	for j in range(len(kmeans.cluster_centers_)):
		#print kmeans.cluster_centers_[j].shape
		#print test_x[i].shape
		dist=compute_distance(kmeans.cluster_centers_[j],test_x[i])
		if (dist < max_d[0]):
			max_d=dist,j
	try:
		if(test_y[i] == lab_d[max_d[1]][0]):
			tmp+=1
		else:
#			print test_y[i],lab_d[max_d[1]][0]
			pass
	except:
#		print lab_d[max_d[1]],max_d[1]
		pass
print tmp*1.0/len(test_x)


#K-means clustering with PCA to reduce dimensionality here

pca=PCA(n_components=25)
pca_np_x = pca.fit_transform(np_x)
#print pca_np_x.shape, np_x.shape
#pca=PCA(n_components=375)
#pca_guess_centers=pca.fit_transform(guess_centers)
#print pca_guess_centers.shape,guess_centers.shape
print "PCA Components", pca.n_components_
loaded_centers=pickle.load(open('centers.p','r'))

kmeans = KMeans(n_clusters=62,max_iter=5000000,verbose=0)
kmeans.fit_predict(pca_np_x)

pickle.dump(kmeans.cluster_centers_,open('centers.p','w'))
pca=PCA(n_components=25)
pca_test_x = pca.fit_transform(test_x)


tmp=0
for i in range(len(pca_test_x)):
	max_d=100000000,0
	for j in range(len(kmeans.cluster_centers_)):
		#print kmeans.cluster_centers_[j].shape
		#print test_x[i].shape
		dist=compute_distance(kmeans.cluster_centers_[j],pca_test_x[i])
		if (dist < max_d[0]):
			max_d=dist,j
	try:
		if(test_y[i] == lab_d[max_d[1]][0]):
			tmp+=1
		else:
#			print test_y[i],lab_d[max_d[1]][0]
			pass
	except:
#		print lab_d[max_d[1]],max_d[1]
		pass
print tmp*1.0/len(test_x)


labels= {}
for i in kmeans.labels_:
	try:
		labels[i]+=1
	except:
		labels[i]=1
print labels
#Some more clever feature engineering here


#Some Spectral Clustering here

#Spectral CLustering Stuff

#graph=image.img_to_graph(np_x[1])
#print graph
#sc = SpectralClustering(n_clusters=2,assign_lables='kmeans',eigen_solver='arpack')
#sc.fit_predict(graph)
#print sc.labels_
#labels = spectral_clustering(graph,n_clusters=2,eigen_solver='arpack')

#print labels.reshape(20,20)
#plt.matshow(labels.reshape(20,20))
#plt.imshow(np_x[1].reshape(20,20))

#Sobel Edge Detection stuff

#sx=ndimage.sobel(np_x[1].reshape(20,20),axis=0,mode='constant')
#sy=ndimage.sobel(np_x[1].reshape(20,20),axis=1,mode='constant')
#sobel=np.hypot(sx,sy)
#for i in range(20):
#	for j in range(20):
#		if sobel[i][j] > 300:
#			sobel[i][j] = 1
#		else:
#			sobel[i][j] = 0
#plt.imshow(sobel)
#print sobel


#simple binarization
min_dist=1000000000,'none'

#avg_img[119]=[0 for x in range(400)]

#print np.asarray(avg_img[48]).reshape(20,20)

#acc=0
#for i in range(len(test_x)):
#	x=find_min_dist(np.asarray(simple_binarization(test_x[i])))
#	if x[1]==chr(test_y[i]):
#		acc+=1
#print acc*1.0 / len(test_x)


spectral_clustering(np_x,np_y)
#print pic
for pic in avg_img.keys():
	plt.matshow(avg_img[pic].reshape(20,20))
plt.show()

#print len(labels)
#print labels

