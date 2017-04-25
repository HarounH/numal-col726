import sys,os,pdb;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sn;
from sklearn.manifold import TSNE;

class CmdOptions:
	def __init__(self):
		self.rank = 5

def main(nd=25, nt=25, rank=5):
	# dtm = np.zeros(nd, nt)
	np.random.seed(7)

	dtm = np.random.choice([0,1], size=(nd,nt), p=[4.0/5,1.0/5])

	u,s,v = np.linalg.svd(dtm)
	# pdb.set_trace()
	svd_dtm = np.dot(np.dot(u[:,:rank], np.diag(s[:rank])), v[:rank,:])
	
	q,r = np.linalg.qr(dtm)
	qr_dtm = np.dot(q[:,:rank], r[:rank,:])
	
	xax = 7
	yax = 4
	# Make three plots and scatter.
	f,axarr = plt.subplots(nrows=1, ncols=3)

	axarr[0].scatter(dtm[:,xax], dtm[:,yax])
	for i in range(0, len(dtm[:,0])):
		axarr[0].annotate(str(i), (dtm[i,xax], dtm[i,yax]))
	axarr[0].set_title('Original')

	axarr[1].scatter(svd_dtm[:,xax], svd_dtm[:,yax])
	for i in range(0, len(svd_dtm[:,0])):
		axarr[1].annotate(str(i), (svd_dtm[i,xax], svd_dtm[i,yax]))
	axarr[1].set_title('SVD')

	axarr[2].scatter(qr_dtm[:,xax], qr_dtm[:,yax])
	for i in range(0, len(qr_dtm[:,0])):
		axarr[2].annotate(str(i), (qr_dtm[i,xax], qr_dtm[i,yax]))
	axarr[2].set_title('QR')

	plt.show();

	return;


	fig, axarr = plt.subplots(3)
	plt.suptitle('rank=' + str(rank))
	cax = []
	cax.append(axarr[0].matshow(dtm, cmap=plt.cm.Blues))
	axarr[0].set_ylabel('Original')

	cax.append(axarr[1].matshow(svd_dtm, cmap=plt.cm.Blues))
	axarr[1].set_ylabel('SVD')

	cax.append(axarr[2].matshow(qr_dtm, cmap=plt.cm.Blues))
	axarr[2].set_ylabel('QR')

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(cax[0], cax=cbar_ax)

	plt.show()

if __name__ == '__main__':
	options = CmdOptions()
	if len(sys.argv)>1:
		options.rank = int(sys.argv[1])
	main(rank=options.rank)
