import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

xtrain=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\train\\X_train.txt',delim_whitespace=True,header=None)
xtest=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\test\\X_test.txt',delim_whitespace=True,header=None)
ytrain=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\train\\y_train.txt',header=None)
ytest=pd.read_table('G:\UF\Sem2\PR\project\UCI HAR Dataset\UCI HAR Dataset\\test\\y_test.txt',header=None)
xtrain=xtrain.values 
xtest=xtest.values
ytrain=ytrain.values
ytest=ytest.values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
############------------> p=1 Manhattan distance <-------------###########
scores = []
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = i,p=1, n_jobs = -1)
    knn.fit(xtrain, ytrain)
    ypred = knn.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')
xticks = range(1,50)
plt.plot(xticks, scores, color='red', linestyle='solid', marker='o',
         markerfacecolor='blue', markersize=5)
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
plt.show()
scores = np.array(scores)
print'Optimal No. Of Neighbors: ', scores.argmax()+1
print'Accuracy Score: '+ str(scores.max()*100)+ ' %'

knn = KNeighborsClassifier(n_neighbors = scores.argmax()+1,p=1, n_jobs = -1)
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)


from sklearn.metrics import confusion_matrix
cma1=confusion_matrix(ypred,ytest)
print cma1
#####----------------> p=2 <--------------#####
scores = []
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = i,p=2, n_jobs = -1)
    knn.fit(xtrain, ytrain)
    ypred = knn.predict(xtest)
    scores.append(accuracy_score(ytest,ypred))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')
xticks = range(1,50)
plt.plot(xticks, scores, color='red', linestyle='solid', marker='o',
         markerfacecolor='blue', markersize=5)
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
plt.show()
scores = np.array(scores)
print'Optimal No. Of Neighbors: ', scores.argmax()+1
print'Accuracy Score: '+ str(scores.max()*100)+ ' %'

knn = KNeighborsClassifier(n_neighbors = scores.argmax()+1, n_jobs = -1)
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)

from sklearn.metrics import confusion_matrix
cma2=confusion_matrix(ypred,ytest)
print cma2


######----------->For Confusion Matrix Plot<-------------######
cma=cma1
import matplotlib.pyplot as plt
import numpy as np
import itertools

accuracy = np.trace(cma) / float(np.sum(cma))
misclass = 1 - accuracy
normalize    = False
target_names = ['1', '2', '3','4','5','6']
title        = "Confusion Matrix"
cmap=None
if cmap is None:
   cmap = plt.get_cmap('Blues')

plt.figure(figsize=(8, 6))
plt.imshow(cma, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()

if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

if normalize:
        cma = cma.astype('float') / cma.sum(axis=1)[:, np.newaxis]


thresh = cma.max() / 1.5 if normalize else cma.max() / 2
for i, j in itertools.product(range(cma.shape[0]), range(cma.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cma[i, j]),
                     horizontalalignment="center",
                     color="white" if cma[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cma[i, j]),
                     horizontalalignment="center",
                     color="white" if cma[i, j] > thresh else "black")


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
plt.show()

#Optimal No. Of Neighbors:  8
#Accuracy Score: 90.7363420428 %
