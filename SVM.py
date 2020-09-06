# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:37:52 2020

@author: jyotik
"""

import pandas as pd
import numpy as npy
from sklearn import svm
from sklearn.model_selection import GridSearchCV



xtrain=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\train\\X_train.txt',delim_whitespace=True,header=None)
xtest=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\test\\X_test.txt',delim_whitespace=True,header=None)
ytrain=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\train\\y_train.txt',header=None)
ytest=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\test\\y_test.txt',header=None)
#from sklearn.decomposition import FastICA

#ica = FastICA(n_components=3)
#xtrain = ica.fit_transform(xtrain)
#xtest = ica.fit_transform(xtest)
clfr=svm.SVC()

param=[{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

model=GridSearchCV(clfr,param,n_jobs=-1,cv=4,verbose=4)
model.fit(xtrain.as_matrix(),ytrain.as_matrix().ravel().T)
#model.fit(xtrain,ytrain)

from sklearn.metrics import accuracy_score
ypred=model.predict(xtest)
acry=accuracy_score(ytest,ypred)

print 'Best Parameters: '+ str(model.best_params_)
print 'Accuracy Score: '+ str(acry*100) + ' %'
from sklearn.metrics import confusion_matrix
cma1=confusion_matrix(ypred,ytest)
print cma1

import matplotlib.pyplot as plot
import itertools

acrcy = npy.trace(cma1) / float(npy.sum(cma1))
mclass = 1 - acrcy
nz    = False
columns = ['1', '2', '3','4','5','6']
titlename        = "Confusion Matrix"
cmap1=None
if cmap1 is None:
   cmap1 = plot.get_cmap('Blues')

plot.figure(figsize=(10, 8))
plot.imshow(cma1, interpolation='nearest', cmap=cmap1)
plot.title(titlename)
plot.colorbar()

if columns is not None:
        t_mks = npy.arange(len(columns))
        plot.xticks(t_mks, columns, rotation=45)
        plot.yticks(t_mks, columns)

if nz:
        cma1 = cma1.astype('float') / cma1.sum(axis=1)[:, npy.newaxis]


trsh = cma1.max() / 1.5 if nz else cma1.max() / 2
for a, b in itertools.product(range(cma1.shape[0]), range(cma1.shape[1])):
        if nz==True:
            plot.text(b, a, "{:0.4f}".format(cma1[a, b]),
                     color="white" if cma1[a, b] > trsh else "black",horizontalalignment="center")
        else:
            plot.text(b, a, "{:,}".format(cma1[a, b]),
                     color="white" if cma1[a, b] > trsh else "black",horizontalalignment="center")


plot.tight_layout()
plot.ylabel('True label')
plot.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acrcy, mclass))
plot.show()
