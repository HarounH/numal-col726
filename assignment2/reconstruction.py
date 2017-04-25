from __future__ import print_function
import sys,os,pdb
import numpy as np
import scipy
from scipy import linalg, matrix
'''
	Points are represented in 3D coordinates. not homogenous.

	In this code, we write
		A_i - A_0 as dA[i] # 1<=i<=n
		a_i - a_0 as da[i] # 1<=i<=n
		a_i^' - a_0^' as dap[i] # 1<=i<=n

	The reconstruction is done using dA[1:4],da[1:4],dap[1:4]

	Beloved system:
		
		Let L = [
			[ (a_1 - a_0) (a_2 - a_0) (a_3 - a_0) ]
			[ (ap_1 - ap_0) (ap_2 - ap_0) (ap_3 - ap_0) ]
		]

		L*(A_k-A_0) = [ [a_k-a_0],[ap_k-ap_0] ]

		Note that the basis for the above system is \( (Ai-A0)_{i=1}^4 \)


	np.matrix()[0] returns a row.
	We want each row to a be a point.

	Hence, each point is a row. That changes things.

	Convert to numpy space by taking transpose.
	(vec_A-vec_A0) * L.T = vec_a-vec_a0 | vec_ap-vec_ap0

	BUT! numpy.linalg.solve(a,b) solves ax=b for x ... so we'll have to transpose when we pass it in anyway.

	So...hmm... well, gonna have to continue to use columns as data points
'''

'''
	This class stores things in standard basis. conversions are made if necessary
	s2c and c2s are transformation matrices.

'''
class MinorQ6:
	'''
		Input: mat_A ... 4 points, A0, A1, A2, A3 such that mat_A[:,k] = point
	'''
	def __init__(self, mat_A, mat_a, mat_ap):
		

		# Save stuff.
		self.mat_A = mat_A
		self.mat_a = mat_a
		self.mat_ap = mat_ap
		# Initial points.
		self.vec_A0 = self.mat_A[:,0] # column vectors.
		self.vec_a0 = self.mat_a[:,0]
		self.vec_ap0= self.mat_ap[:,0]

		# Alter stuff
		self.mat_A -= self.vec_A0
		self.mat_a -= self.vec_a0
		self.mat_ap -= self.vec_ap0

		# Basis change matrix.


		# TODO Catch invalid points here
		# BC*(A1 - A0) = column([1,0,0]) etc
		# => BC = ([ (A1-A0) (A2-A0) (A3-A0) ]).inv
		self.s2c = np.linalg.inv( mat_A[:,1:4] ) # Converts basis from standard 3D to custom
		self.c2s = np.linalg.inv(self.s2c) # Converts basis from custom to 3D

		# Setup transformation matrix.
		self.L = np.concatenate((self.mat_a[:,1:4], self.mat_ap[:,1:4]))
		self.Linv = np.linalg.pinv(self.L)
	'''
		Input: A , a point in 3D.
		Output: a, ap... corresponding points.
	'''
	def project(self, vec_A):
		temp = self.L.dot(self.s2c.dot(vec_A - self.vec_A0))
		return temp[:,0:2], temp[:,2:4]

	'''
		Takes in vectors vec_a, vec_ap in arbit representation and returns the corresponding point in basis of mat_A
	'''	
	def unproject(self, vec_a, vec_ap):
		target = np.concatenate(( vec_a - self.vec_a0, vec_ap - self.vec_ap0))
		return self.vec_A0 + self.c2s.dot(self.Linv.dot(target))

def main():
	A0 = np.matrix([0,0,0]).T
	a0 = np.matrix([0,0]).T
	ap0 = np.matrix([0,0]).T
	
	A1 = np.matrix([2,1,0]).T
	a1 = np.matrix([2,1]).T
	ap1 = np.matrix([1,0]).T
	
	A2 = np.matrix([1,2,0]).T
	a2 = np.matrix([1,2]).T
	ap2 = np.matrix([1,0]).T
	
	A3 = np.matrix([2,2,-2]).T
	a3 = np.matrix([2,2]).T
	ap3 = np.matrix([2,-2]).T
	
	mat_A = np.concatenate((A0,A1,A2,A3), axis=1)
	mat_a = np.concatenate((a0,a1,a2,a3), axis=1)
	mat_ap = np.concatenate((ap0,ap1,ap2,ap3), axis=1)

	solution = MinorQ6(mat_A, mat_a, mat_ap)
	print(solution.project(np.matrix([0,0,-10]).T))
	print(solution.unproject( np.matrix([0,0]).T, np.matrix([0,-10]).T))

if __name__ == '__main__':
	main()