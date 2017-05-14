"""
http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html
A recursive feature elimination example showing the relevance
of pixels in a digit classification task.

Programmer:wuyi
Location:wuhan
Date:2017-05-10
"""
print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

#load the digits dataset
digits = load_digits()
x = digits.images.reshape(len(digits.images),-1)
y = digits.target

#Create the RFE object and rank each pixel
svc = SVC(kernel="linear",C=1)
rfe = RFE(estimator=svc,n_features_to_select=1,step=1)
rfe.fit(x,y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

#Plot pixel ranking
plt.matshow(ranking,cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()