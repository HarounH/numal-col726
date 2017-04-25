import sys,os,pdb;
import numpy as np;
import matplotlib.pyplot as plt;
import _pickle as pickle;
from PIL import Image;
from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.svm import SVC;
from sklearn.metrics import accuracy_score, confusion_matrix;

class CmdOptions:
	def __init__(self, argv):
		if len(argv)>1:
			self.n_components = int(sys.argv[1])
		else:
			self.n_components = 5

		if len(argv)>3:
			self.train = argv[2]=='train'
			self.model_file = argv[3]
		else:
			self.train = True
			self.model_file = 'clf.bin'
def load_data():
	'''
		@returns trainX,trainY,testX,testY, scaler
	'''
	directory = 'orl_faces/'
	n_subjects = 40
	n_images = 10
	trainX = []
	testX = []
	trainY = []
	testY = []
	for ns in range(1, n_subjects+1):
		for ni in range(1, n_images+1):
			infilename = directory + 's' + str(ns) + '/' + str(ni) + '.png'
			img = Image.open(infilename)
			vec = np.array(img).ravel()
			# pdb.set_trace()
			if ni>5:
				testX.append(vec)
				testY.append(ns)
			else:
				trainX.append(vec)
				trainY.append(ns)
			img.close()
	trainX = np.array(trainX)
	trainY = np.array(trainY)
	testX = np.array(testX)
	testY = np.array(testY)

	scaler = StandardScaler()
	trainX = scaler.fit_transform(trainX)
	testX = scaler.transform(testX)
	
	# pdb.set_trace()
	return trainX,trainY,testX,testY, scaler

def get_pca(X, n_components):
	'''
		@in X : design matrix
		@in n_components : number of eigen faces to use

		@return pca : pca model

		pca.components_ should give you the eigenfaces.
	'''
	pca = PCA(n_components=n_components)
	pca.fit(X)
	return pca

def train(X, y):
	'''
		@in X a matrix whose rows are datapoints ... design matrix
		@in y corresponding classes for each row of X. Not OneHotEncoded.
		@return model a simple SVM :)
	'''
	model = SVC()
	model.fit(X,y)
	return model

def plot_cnf_mat(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, cmap=cmap)
	plt.title(title)
	plt.colorbar()
	plt.xlabel('Prediction')
	plt.ylabel('Gold')
	plt.show()

def show_face(vec, nx=112, ny=92, title=None):
	plt.imshow( np.reshape(vec, [nx,ny]), cmap='gray' )
	if title is not None:
		plt.title(title)
	plt.show()

def visualizations(trainX, testX, pca, scaler):
	'''
		A function to look at things in the PCA, such as components,
		to plot their eigen values etc.
	'''
	# pdb.set_trace()
	all_data = np.concatenate([scaler.inverse_transform(trainX),scaler.inverse_transform(testX)])
	
	# Average face
	# show_face(all_data.mean(axis=0),title='Avg Face')

	# Eigen faces
	# for i in range(0, 10):
	# 	show_face(pca.components_[i],title='Eigen face ' + str(i))
	f,axarr = plt.subplots(2,5)
	for i in range(0, 10):
		axarr[i%2,int(i/2.0)].imshow(np.reshape(pca.components_[i],[112,92]), cmap='gray')
		axarr[i%2,int(i/2.0)].get_xaxis().set_visible(False)
		axarr[i%2,int(i/2.0)].get_yaxis().set_visible(False)
	plt.show()
	# Weird faces
	# show_face(all_data[0] + 0.5*all_data[11], title='interesting face')

	pass

def main(options):
	trainX,trainY,testX,testY,scaler = load_data()
	print('Data loading done')
	pca = get_pca(trainX, options.n_components)
	visualizations(trainX, testX, pca, scaler)
	print('Got PCA')
	if options.train==True:
		model = train(pca.transform(trainX), trainY)
		with open(options.model_file, 'wb') as mf:
			pickle.dump(model, mf)
	else:
		with open(options.model_file, 'rb') as mf:
			model = load_model(mf)
	print('Trained model')
	predictedY = model.predict(pca.transform(testX))
	print('Prediction complete')
	# Get accuracy and stuff
	cnf_mat = confusion_matrix(testY, predictedY)
	acc = accuracy_score(testY, predictedY)
	print('Accuracy=',acc)
	plot_cnf_mat(cnf_mat)
	return
	n_components_list=[2,5,7,10,15,20,25,30,40,50,75,100,250,500,750,1000]
	accuracies = []
	for n_components in n_components_list:
		pca = get_pca(trainX, n_components)
		model = train(pca.transform(trainX), trainY)
		predictedY = model.predict(pca.transform(testX))
		accuracies.append(accuracy_score(testY, predictedY))
	plt.plot(n_components_list, accuracies)
	plt.title('Accuracy vs Rank')
	plt.xlabel('Rank')
	plt.ylabel('Accuracy')
	plt.show()
if __name__ == '__main__':
	options = CmdOptions(sys.argv)
	main(options)
