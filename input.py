import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread_collection,imshow,concatenate_images,imread
from skimage.transform import resize
from skimage.color import rgb2gray
import math
from sklearn import svm
from sklearn.neural_network import MLPClassifier
	
print "Collecting images..."
input_images=[]
output_images=imread_collection('segmentation/Train_Data/*-mask.jpg')[0:164]
for i in xrange(164):
	input_images.append(imread('segmentation/Train_Data/train-'+str(i)+'.jpg'))
x=[]
y=[]
mlp = MLPClassifier(hidden_layer_sizes=(50,100,),verbose=True)
X=[]

for k in range(41):
# For each image
	print k
	img_gray = rgb2gray(input_images[k])
	y = np.array(output_images[k]).flatten()
	dist=np.empty([128,128])
	x=np.empty([128,128,2])
	n=math.sqrt(((0-63.5)**2)+((0-63.5)**2))
	for i in xrange(len(dist)):
		for j in xrange(len(dist)):
		# For each pixel
			dist[i][j]=math.sqrt(((i-63.5)**2)+((j-63.5)**2))
			#print n,"eee"
			#print dist[i][j],"ddd"
			dist[i][j]=dist[i][j]/n
			x[i][j][0]=dist[i][j]
			x[i][j][1]=img_gray[i][j]
			
			
	x=np.reshape(x,(len(x)*len(x),2))
	print x
	print y
	 	
	mlp.fit(x, y)
	#x=x.flatten()
	#X.append(x)

for k in range(41,82):
# For each image
	print k
	img_gray = rgb2gray(input_images[k])
	y = np.array(output_images[k]).flatten()
	dist=np.empty([128,128])
	x=np.empty([128,128,2])
	n=math.sqrt(((0-63.5)**2)+((0-63.5)**2))
	for i in xrange(len(dist)):
		for j in xrange(len(dist)):
		# For each pixel
			dist[i][j]=math.sqrt(((i-63.5)**2)+((j-63.5)**2))
			#print n,"eee"
			#print dist[i][j],"ddd"
			dist[i][j]=dist[i][j]/n
			x[i][j][0]=dist[i][j]
			x[i][j][1]=img_gray[i][j]
			
			
	x=np.reshape(x,(len(x)*len(x),2))
	print x
	print y
	p = mlp.predict(x)
	imshow(np.reshape(p,(128,128)))
	plt.show()







print "Done."
