"""
http://scikit-learn.org/stable/auto_examples/feature_selection/feature_selection_pipeline.html
Simple usage of Pipeline that runs successively a univariate feature selection
with anova and then a C-SVM of the selected features.

Programmer:wuyi
Location:wuhan,China
Date:2017-05-10
"""
print(__doc__)

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import make_pipeline

#import some data to play with
x,y = samples_generator.make_classification(n_features=20,n_informative=3,n_redundant=0,n_classes=4,n_clusters_per_class=2)
print x,y,'len:',len(x)
#ANOVA SVM-C
#1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression,k=3)
#2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter,clf)
anova_svm.fit(x,y)
plist = anova_svm.predict(x)

#caculate the correct rate
subList = y-plist
print 'subList:',subList
cNum = 0.0
tNum = len(subList)
for i in range(0,99):
    if subList[i] == 0:
        cNum += 1
print 'tNum:',tNum
per = cNum/tNum
print 'per:',per