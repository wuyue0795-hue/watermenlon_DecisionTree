from untils import *
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
import numpy as np

if __name__=='__main__':
    X = np.array(D.iloc[:,:-1])
    y = np.array(D.iloc[:,-1])
    feature_list = set(X[:,:-2].reshape(-1))
    dt = DecisionTree()
    dt.generateTree(D)
    s2i = {s:i for i,s in enumerate(feature_list)}
    i2s = {i:s for i,s in enumerate(feature_list)}
    for i in range(len(X)):
        for j in range(len(X[i][:-2])):
            X[i][j] = s2i[X[i][j]]
    decisionTree = tree.DecisionTreeClassifier()
    decisionTree.fit(X,y)
    # print(decisionTree.predict(X)==y)
    for i in range(len(D)-1):
        x = D.iloc[i+1,:-1]
        y = D.iloc[i,-1]
        # print(dt.predict(x),y)