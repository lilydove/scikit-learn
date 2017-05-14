"""
Model selecthon
Score, Kfold and Cross-validation generators
"""
from sklearn import datasets,svm
digits = datasets.load_digits()
x_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1,kernel='linear')
print svc.fit(x_digits[:-100],y_digits[:-100]).score(x_digits[-100:],y_digits[-100:])

import numpy as np
x_folds = np.array_split(x_digits,3)
print x_folds
y_folds = np.array_split(y_digits,3)
scores = list()
for k in range(3):
    #We use 'list' to copy,in order to 'pop' later on
    x_train = list(x_folds)
    x_test = x_train.pop(k)
    x_train = np.concatenate(x_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(x_train,y_train).score(x_test,y_test))
print(scores)

#Cross-validation generators
from sklearn.model_selection import KFold,cross_val_score
X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits = 3)
for train_indices,test_indices in k_fold.split(X):
    print('Train:%s | test:%s' %(train_indices,test_indices))

kfold = KFold(n_splits = 3)
print [svc.fit(x_digits[train],y_digits[train]).score(x_digits[test],y_digits[test])
     for train,test in k_fold.split(x_digits)]

print 'cross_val_score:',cross_val_score(svc,x_digits,y_digits,cv=k_fold,scoring='precision_macro')

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets,svm

digits = datasets.load_digits()
x= digits.data
y=digits.target

#exercise:plot the cross-validation score
svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10,0,10)