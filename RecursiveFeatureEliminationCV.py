"""
http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
A recursive feature elimination example
with automatic tuning of the number of features selected
with cross-validation

Programmer:wuyi
Location:wuhan
Date:2017-05-11
"""

print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

#Build a classification task using 3 informative features
x,y = make_classification(n_samples=1000,n_features=25,n_informative=3,
                          n_redundant=2,n_repeated=0,n_classes=8,
                          n_clusters_per_class=1,random_state=0)

#Create the RFE oject and compute a cross-validated score.
svc = SVC(kernel="linear")
#The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=svc,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(x,y)

print("Optimal number of features : %d" % rfecv.n_features_)

#Plot number of feaures VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classification)")
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()
