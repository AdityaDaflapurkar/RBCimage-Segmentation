import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread_collection,imshow,concatenate_images,imread
from skimage.transform import resize
from skimage.color import rgb2gray
import math
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pickle
from scipy import ndimage
	
class Segment(object):

    path = ""

    def __init__(self, data_dir):
        """
            data_directory : path like /home/rajat/nnproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """
	self.model = MLPClassifier(hidden_layer_sizes=(64,32,64),verbose=True)

    def train(self):
        """
            Trains the model on data given in path/train.csv

            No return expected
        """
	print "Collecting images..."
	input_images=[]
	output_images=imread_collection('segmentation/Train_Data/*-mask.jpg')[0:164]
	for i in xrange(164):
		input_images.append(imread('segmentation/Train_Data/train-'+str(i)+'.jpg'))
	x=[]
	y=[]
	# mlp = MLPClassifier(hidden_layer_sizes=(50,100,),verbose=True)
	X=[]

	for k in range(123):
	# For each image
		print k
		img_gray = rgb2gray(input_images[k])
		b = np.ones((6,6))

		av = ndimage.convolve(img_gray, b, mode='reflect')

		y = np.array(output_images[k]).flatten()
		dist=np.empty([128,128])
		x=np.empty([128,128,3])
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
				x[i][j][2]=av[i][j]
			
		x=np.reshape(x,(len(x)*len(x),2))
		print x
		print y
	 	
		self.model.fit(x, y)


    def get_mask(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: A list of lists with the same 2d size as of the input image with either 0 or 1 as each entry

        """

	img_gray = rgb2gray(image)
	#y = np.array(output_images[k]).flatten()
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

	x=np.reshape(x,(len(x)*len(x),3))
	print x
	print y
	p = self.model.predict(x)
	return np.reshape(p,(128,128))
	


    def save_model(self, **params):
        with open('model.pkl', 'wb') as f:
    		pickle.dump(self.model, f)

    @staticmethod
    def load_model(**params):
	seg = Segment("dataset/")
        with open('model.pkl', 'rb') as f:
    		clf = pickle.load(f)
	seg.model = clf
	return seg
	
	

if __name__ == "__main__":
	obj = Segment('dataset/')
	obj.train()
	#obj.save_model(name="segment.gz")
	obj = Segment.load_model()
	print "Collecting images..."
	input_images=[]
	for i in range(1,5):
		input_images.append(imread('segmentation/Train_Data/train-'+str(i)+'.jpg'))
		imshow(obj.get_mask(input_images[i]))
		plt.show()
