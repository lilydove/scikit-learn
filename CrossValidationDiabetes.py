"""
This exercise is used in the Cross-validated estimators part of the Model selection:
choosing estimators and their parameters section of the A tutorial
on statistical-learning for scientific data processing.
programmer:wuyi
location:wuhan
date:2017-05-08
"""
from __future__ import print_function
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

diabetes = datasets.load_diabetes()
x = diabetes.data[:150]
y = diabetes.target[:150]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4,-0.5,30)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso,x,y,cv=n_folds,n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

scores,scores_std = np.array(scores),np.array(scores_std)

plt.figure("Lasso_cv:alpha-scores").set_size_inches(8,6)
plt.semilogx(alphas,scores)

#plot error lines showing +/-std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas,scores + std_error,'b--')
plt.semilogx(alphas,scores - std_error,'b--')

# alphs=0.2 controls the translucency of the fill color
plt.fill_between(alphas,scores + std_error,scores - std_error,alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores),linestyle='--',color='.5')
plt.xlim([alphas[0],alphas[-1]])
plt.show()

#the selection of alpha
lasso_cv = LassoCV(alphas=alphas,random_state=0)
k_fold = KFold(3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k,(train,test) in enumerate(k_fold.split(x,y)):
    lasso_cv.fit(x[train],y[train])
    print("[fold {0} ]alpha:{1:.5f},score:{2:.5f}".format(k,lasso_cv.alpha_,lasso_cv.score(x[test],y[test])))
    print()
    print("Answer: Not very much since we obtained different alphas for different")
    print("subsets of the data and moreover, the scores for these alphas differ")
    print("quite substantially.")
plt.show()
