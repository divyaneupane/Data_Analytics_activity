import numpy as np  
from scipy.signal import convolve2d
#input array
A= np.array([[1,2,3],[5,6,7],[10,0,11]])
B= np.array([[5,3],[9,1]])
#convolution
C= convolve2d(A,B,mode='valid') 
print("Input Array A:\n",A)
print("\nKernel B:\n",B)
# print("\nB flipped both horizontally and vertically:\n",np.flipud(np.fliplr(B)))
print("\nConvolution Result C:\n",C)
print("\nShape of Convolution Result C:",C.shape)

