### 
# Implementation of Multi Layered Perceptron that acts as XOR Gate using 
# Backpropagation Algorithm
###

import sigmoid as s

##define inputs and target for xor gate
x1=[0,0,1,1]		#input1
x2=[0,1,0,1]		#input2
t=[0,1,1,0]  	#target

## Initialize random weights and biases

# Hidden layer first Perceptron
b1=-0.3
w11=0.21
w21= 0.15
# Hidden Layer Second Perceptron
b2=0.25
w12=-0.4
w22=0.1
# Output layer Perceptron
b3=-0.4
w13=-0.2
w23=0.3

error=0
iteration=0
train=True
print("weight are:")
print("w11 : %4.2f w12: %4.2f w21: %4.2f w22: %4.2f w13: %4.2f  w23: %4.2f  \n" %(w11,w12,w21,w22,w13,w23))


## Training Starts

while(train):
	
	for i in range(len(x1)):

		##input for each perceptron of hidden layer
		z_in1=b1+x1[i]*w11+x2[i]*w21
		z_in2=b2+x1[i]*w12+x2[i]*w22
		##computing activation function output
		z1=round(s.sigmoid(z_in1),4)
		z2=round(s.sigmoid(z_in2),4)

		# Output layer forward pass
		y_in=b3+z1*w13+z2*w23
		y=round(s.sigmoid(y_in),4)

		##error computation
		del_k=round((t[i]-y)*y*(1-y),4)
		error=del_k
		##Back pass
		# weight update for output layer
		w13=round(w13+del_k*z1,4)
		w23=round(w23+del_k*z2,4)
		b3=round(b3+del_k,4)

		##error computation for hidden layer
		del_1=del_k*w13*z1*(1-z1)
		del_2=del_k*w23*z2*(1-z2)

		## update weight and biases
		b1=round(b1+del_1,4)
		w11=round(w11+del_1*x1[i],4)
		w12=round(w12+del_1*x1[i],4)

		b2=round(b2+del_2,4)
		w21=round(w21+del_2*x2[i],4)
		w22=round(w22+del_2*x2[i],4)

		print("Iteration: ",iteration)
		print("w11 : %5.4f w12: %5.4f w21: %5.4f w22: %5.4f w13: %5.4f  w23: %5.4f " %(w11,w12,w21,w22,w13,w23))
		print("Error: %5.3f" %del_k)
		iteration=iteration+1

	if(abs(error)<=0.105):
		train=False 







