from __future__ import print_function
import sys,os,pdb
import numpy as np
# from sympy import Matrix
import scipy
from scipy import linalg, matrix

eps = 1e-12

'''
	Given a matrix, of some dimension, commputes the basis of nullspace.
	Stolen from http://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
'''
def null(A):
	global eps
	u, s, vh = scipy.linalg.svd(A)
	padding = max(0,np.shape(A)[1]-np.shape(s)[0]) # need to make it a square matrix.
	null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
	null_space = scipy.compress(null_mask, vh, axis=0)
	return scipy.transpose(null_space)

'''
	Given a matrix, of some dimension, commputes the basis of rangespace.
'''
def rangespace(A):
	global eps
	u, s, vh = scipy.linalg.svd(A)
	padding = max(0,np.shape(A)[1]-np.shape(s)[0]) # need to make it a square matrix.
	null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
	range_space = scipy.compress(np.logical_not(null_mask), vh, axis=0)
	return scipy.transpose(range_space)

'''
	Given a matrix, it computes QR decomposition
	returns:
		nullspace of A,
		nullspace of A as nullspace of R (Q has fullrank, it is invertible)
		Q*nullspace(R)
		rangespace of A
		rangespace of Q, rangespace of R
	Doesn't care for transpose.
'''
def solve_question(A):
	Q,R = linalg.qr(A, pivoting=False)
	nr = null(R)
	qnr = Q.dot(nr)
	return null(A), nr, qnr, rangespace(A), rangespace(Q), rangespace(R)

if __name__ == '__main__':
	
	A = matrix([ [1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16] ])
	na,nr, qnr, ra, rq, rr = solve_question(A)
	print('Nullspaces')
	print('null(A)\n' + str(na))
	print('null(R)\n' + str(nr))
	print('Q*null(R)\n' + str(qnr))
	print('Rangespaces')
	print('Rangespace(a)\n' + str(ra))
	print('Rangespace(q)\n' + str(rq))
	print('Rangespace(r)\n' + str(rr))
	
	print('Matrix rank of na augmented with nr')
	test = matrix([ na.T[0], na.T[1], nr.T[0], nr.T[1] ])
	print(test)
	print(np.linalg.matrix_rank(test))