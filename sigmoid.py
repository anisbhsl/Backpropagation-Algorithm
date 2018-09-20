import math

def sigmoid(x):
	'''
	Sigmoid Activation Function
	'''
	y= 1/(1+math.exp(-x))
	return y 
