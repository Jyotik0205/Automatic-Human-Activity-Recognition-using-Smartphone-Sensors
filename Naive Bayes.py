# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:18:37 2020

@author: jyotik
"""

import numpy as np
import pandas as pd

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
xtrain=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\train\\X_train.txt',delim_whitespace=True,header=None)
xtest=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\test\\X_test.txt',delim_whitespace=True,header=None)
ytrain=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\train\\y_train.txt',header=None)
ytest=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\test\\y_test.txt',header=None)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(xtrain, ytrain)
GaussianNB()
ypred=clf.predict(xtest)
yprob=clf.predict_proba(xtest)

from sklearn.metrics import accuracy_score
for i in range(ypred.size):
 if ypred[i]!=ytest[0][i]:
    print str(ypred[i])+'at'+str(i)
    print ytest[0][i]
    print yprob[i]
accuracy=accuracy_score(ytest,ypred)
print 'Accuracy Score: '+ str(accuracy*100) + ' %'

# clf_pf = GaussianNB()
#clf_pf.partial_fit(X, Y, np.unique(Y))
#GaussianNB()
