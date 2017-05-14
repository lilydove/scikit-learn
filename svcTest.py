__author__ = 'wuyi'
# using svm of scikit-learn to predict
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

print len(digits.data)
print digits.data
clf = svm.SVC(gamma=0.001,C=100.)
clf.fit(digits.data[:-3],digits.target[:-3])#training sets
rs = clf.predict(digits.data[-3:])

print rs