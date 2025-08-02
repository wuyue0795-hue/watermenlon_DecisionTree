import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
import sklearn.datasets as datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sklearn.tree as tree
import os


bcDataset = datasets.load_breast_cancer()

x = bcDataset.data
y = bcDataset.target
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.2)

entropy_thresholds = np.linspace(0,1,50)
gini_thresholds = np.linspace(0,0.5,50)
param_grid = [
    # {'criterion':['entropy'],'min_impurity_decrease': entropy_thresholds},
    # {'criterion':['gini'],'min_impurity_decrease': gini_thresholds},
    # {'max_depth':range(2,10)},
    {'min_samples_split':range(2,30,2)}]

clf = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,cv=5)
clf.fit(x,y)
print('best param is {0}, best score is {1}'.format(clf.best_params_,clf.best_score_))

clf = clf.best_estimator_
print(clf)
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)
print('resolved score is {0}'.format(score))
with open('breast_cancer.dot','w') as f:
    f = tree.export_graphviz(clf,out_file=f)
