'''
	Code for numerical algorithms - assignment 1.
	@authors Haroun H
'''
from __future__ import print_function
import sys,os,pdb
from matplotlib import pyplot as plt
import numpy as np
'''
	@params 
		f : Unary function that takes a float input and returns a float
		a,b: lower and upper range of integration
		h : step size for integration
	@returns 
		ans: integral of f from a to b
'''
def trapezoidal_rule(f, a, b, h):
	# Doing it as stably as possible
	xl = a
	xu = xl + h
	ans = 0
	while xu<b:
		ans += h*(f(xl) + f(xu)) # Multiply by h here to avoid overflow?
		xl += h
		xu += h
	ans *= 0.5
	return ans

'''
	@params
		f : fn: float*float -> float
		n : number of values at step size of h
		h : step size
		y0: boundary condition for y at 0
	@return 
		x,y : sequence of function values at points in the range(0:nh:h)
'''
def euler_method(f, n, h, y0):
	x = []
	y = []
	xk = 0
	yk = y0
	for k in range(0,n):
		x.append(xk)
		y.append(yk)
		yk += h*f(xk, yk)
		xk += h
	return x,y

'''	
	Newton's method for computing sqrt(2)

'''
def newton_method():
	correct_value = 1.41421356237
	mdoa = 12 # maximum number of digits of accuracy
	ndoa = 0 # Number of digits of accuracy
	xk = 1
	k = 0
	ns = []
	ks = []
	xks = [xk]
	while ndoa<mdoa:
		k += 1
		xk = 0.5*(xk + (2.0/xk))
		xks.append(xk)
		# Figure out how many digits of accuracy we have
		while (abs(xk - correct_value) < (10**(1-ndoa))):
			print(str(ndoa) + ' digits of accuracy in ' + str(k) + ' iterations')
			ndoa += 1
		ns.append(ndoa)
		ks.append(k)

	print('calculated value:' + str(xk))
	plt.scatter(ns, ks, label='observations (in steps)', color='blue')
	plt.scatter(ns, [np.exp(k) for k in ks], label='e^observations', color='red')
	m,b = np.polyfit(ns, [np.exp(k) for k in ks], 1)
	plt.plot(ns, [m*n + b for n in ns], label='line through e^observations', color='red')
	# plt.plot([ns[0], ns[-1]], [np.exp(ks[0]), np.exp(ks[-1])], label='straight line through exp(observations)', color='green')
	
	plt.ylabel('#steps')
	plt.title('Number of steps vs number of digits of accuracy')
	plt.legend(loc='upper left')
	plt.plot()
	plt.show()

	plt.scatter(ns, ks, label='observations', color='blue')
	# plt.plot([ns[0], ns[-1]], [np.exp(ks[0]), np.exp(ks[-1])], label='straight line through exp(observations)', color='green')
	plt.xlabel('#digits of accuracy')
	plt.ylabel('#steps')
	plt.title('Number of steps vs number of digits of accuracy')
	plt.legend(loc='best')
	plt.plot()
	plt.show()

	
def pi_instability():
	f = lambda y0: lambda x: np.pi*y0/(y0 + (np.pi - y0)*(np.exp(x**2)))
	xs = np.linspace(0,10,100)
	pi8 = np.round(np.pi, 8)
	pi9 = np.round(np.pi, 9)
	f8 = f(pi8)
	f9 = f(pi9)
	y_pi8s = [f8(x) for x in xs]
	y_pi9s = [f9(x) for x in xs]
	y_true = [np.pi for x in xs]
	plt.plot(xs,y_pi8s,label='8 digits of np.pi')
	plt.plot(xs,y_pi9s,label='9 digits of np.pi')
	plt.plot(xs,y_true,label='all digits of np.pi')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('y vs x')
	plt.legend(loc='best')
	plt.show()

def qutb_minar_to_gurugram():
	b = np.array([1,2])
	a0 = np.array([[2,-4],[-2.998, 6.001]])
	a1 = np.array([[2,-4],[-2.998, 6.0]])
	s0 = np.linalg.solve(a0,b)
	s1 = np.linalg.solve(a1,b)
	print('s0:' + str(s0))
	print('s1:' + str(s1))


def polynomial():
	print('original roots:' + str(np.roots([1, -102, 201, -100])))
	print('original roots:' + str(np.roots([1, -102, 200, -100])))
	def evalpolynomial(coeffs, x):
		ans = 0
		n = len(coeffs)
		for i in range(0, len(coeffs)):
			ans += coeffs[i]*(x**(n-i))
		return ans
	print('value at 1:' + str(evalpolynomial([1, -102, 201, -100], 1.0)))
	print('value at 1:' + str(evalpolynomial([1, -102, 200, -100], 1.0)))

def analyse_root_formula():
	def quadratic_roots(a,b,c):
		normal0 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
		normal1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
		new = -2*c/(b**2 + np.sqrt(b**2 -4*a*c))
		return normal0, normal1, new
	a = 1
	c = 1
	blo = 10
	bhi = 1000
	bstep = 1
	bs = []
	roots0 = []
	roots1 = []
	rootsn = []

	for b in range(blo, bhi, bstep):
		r0,r1,rn = quadratic_roots(a,b,c)
		roots0.append(r0)
		roots1.append(r1)
		rootsn.append(rn)
		bs.append(b)
	plt.plot(bs, roots0, label='usual formula')
	# plt.plot(bs, roots1, label='roots1')
	plt.plot(bs, rootsn, label='new formula')
	plt.legend(loc='best')
	plt.xlabel('b')
	plt.ylabel('root')
	plt.title('Roots v/s b for a=' + str(a) + ', c=' + str(c))
	plt.axis([blo-100, bhi+100, min(min(roots0) , min(rootsn)), -1*min(min(roots0) , min(rootsn))])
	plt.show()

def integration_by_parts():
	I0 = 1 - (1.0/np.e)
	sequence1 = [I0]
	sequence2 = [I0]
	for k in range(1, 21):
		print('k=' + str(k))
		Ik = 0
		if k%2==0: # update sequence2
			Ik = (1.0/np.pi) - (k*(k-1)/(np.pi**2))*sequence2[-1]
			sequence2.append(Ik)
		# update sequence 1 anyway
		Ik = 1.0 - k*sequence1[-1]
		sequence1.append(Ik)
	print('Sequence1:')
	for ik in sequence1:
		print(ik)
	print('Sequence2:')
	for ik in sequence2:
		print(ik)

	plt.plot([0.0,20.0], [0.0, 0.0], color='black')
	plt.plot([0.0,20.0], [1.0, 1.0], color='black')
	plt.plot([0.0,20.0], [-1.0, -1.0], color='black')
	# plt.plot(list(range(0,21)), sequence1, label='step size 1')
	plt.plot(list(range(0,21,2)), sequence2, label='step size 2')
	plt.xlabel('k')
	plt.ylabel('Ik')
	plt.title('Ik vs k')
	plt.legend(loc='best')
	plt.axis([0, 20, -5.0, 5.0])
	plt.show()

def deviation_analysis(xs):
	xbar = np.mean(xs)
	n = len(xs)
	s0 = 0
	s1 = 0
	for x in xs:
		s0 += (x-xbar)**2
		s1 += (x**2) - (xbar**2)
	s0 /= n
	s1 /= n
	print('for ' + str(xs))
	print('mu:' + str(xbar))
	print('s0: ' + str(s0))
	print('s1: ' + str(s1))

def main():
	# Tut1: Part1 Q1 - integrate sin x using trapezoidal rule.
	# for h in [0.1, 0.01]:
	# 	print('h=' + str(h) + ' integral=' + str(trapezoidal_rule(np.sin, 0, np.pi, h)))
	
	# Tut1: Part1 Q2c - solve differential equation
	# x,yapprox = euler_method((lambda xk,yk: (2*xk*yk - 2*xk*xk + 1)), 10, 0.1, 1)
	# yexact = [ (np.exp(xk**2) + xk) for xk in x ]
	# plt.plot(x, yapprox, label='Approximation by Euler')
	# plt.plot(x, yexact, label='using solution to DE')
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.title('Experiment with differential equation')
	# plt.legend(loc='best')
	# plt.show()

	# Tut1: Part1 Q3c
	# newton_method()

	# Tut1: Part2 Q1
	# pi_instability()

	# Tut1: Part2 Q1
	# qutb_minar_to_gurugram()

	# Tut1: Part2 Q3
	# analyse_root_formula()

	#Tut1: Part3 Q3
	# integration_by_parts()

	#Tut1: Part3 Q4
	deviation_analysis([1-11e-12] + ([1.0+1e-12]*9))

if __name__ == '__main__':
	main()