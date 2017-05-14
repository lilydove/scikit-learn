from sklearn import *
iris = datasets.load_iris()
data = iris.data
print data.shape    #150,4:150 items and 4 features
#print iris.DESCR

#analysis the digits dataset
digits = datasets.load_digits()
aTuple= digits.images.shape #1797 8*8
print aTuple
import matplotlib.pyplot as plt
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r)
#transform each 8*8 image into a feature vector of length 64
data = digits.images.reshape((digits.images.shape[0],-1))
