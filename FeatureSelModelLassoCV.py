"""
http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html
Use SelectFromModel meta-transformer along with Lasso to select the best couple of features from the Boston dataset.

Programmer:wuyi,Manoj Kumar<mks542@nyu.edu>
Location:wuhan
Date:2017-05-12
License: BSD 3 clause

"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

#Load the boston dataset.
boston = load_boston()
print boston.DESCR
x,y = boston['data'],boston['target']

#We use the base estimator LassoCV  since the L1 norm promotes sparsity of features
clf = LassoCV()

#Set a minimun threshold of 0.25
sfm = SelectFromModel(clf,threshold=0.25)
sfm.fit(x,y)
n_features = sfm.transform(x).shape[1]

#Reset the threshold till the number of features equals two.
#Note that the attribute can be ser directly instead of repeatedly
#fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    x_transform = sfm.transform(x)
    n_features = x_transform.shape[1]

#Plot the selected two features from x.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = x_transform[:,0]
feature2 = x_transform[:,1]
plt.plot(feature1,feature2,'r.')
plt.xlabel("Feature number1")
plt.ylabel("Feature number2")
plt.ylim([np.min(feature2),np.max(feature2)])
plt.show()