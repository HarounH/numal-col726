import sys,os,pdb;
import numpy as np;
from PIL import Image
def main():
	'''
		Reads the images in the folder and simply converts them to png instead of pgm. 
		I want to be able to visualise these images.
	'''
	directory = 'orl_faces/'
	n_subjects = 40
	n_images = 10
	for ns in range(1, n_subjects+1):
		for ni in range(1, n_images+1):
			infilename = directory + 's' + str(ns) + '/' + str(ni) + '.pgm'
			outfilename= directory + 's' + str(ns) + '/' + str(ni) + '.png'
			img = Image.open(infilename)
			img.save(outfilename)

if __name__ == '__main__':
	main()