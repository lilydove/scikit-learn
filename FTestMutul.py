"""
http://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html
This example illustrates the differences between univariate F-test statistics and mutual information.
Programmer:wuyi
Location:wuhan
Date:2017-05-11
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression,mutual_info_regression

np.random.seed(0)
x = np.random.rand(1000,3)
y = x[:,0] + np.sin(6*np.pi*x[:,1]) + 0.1 * np.random.rand(1000)

f_test,_ = f_regression(x,y)
f_test /= np.max(f_test)

mi = mutual_info_regression(x,y)
mi /= np.max(mi)

plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.scatter(x[:,i],y)
    plt.xlabel("$x_{}$".format(i+1),fontsize=14)
    if i == 0:
        plt.ylabel("$y$",fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i],mi[i]),fontsize=16)
plt.show()
