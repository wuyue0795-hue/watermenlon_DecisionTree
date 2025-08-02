import pandas as pd
import numpy as np
import math
from tqdm import tqdm


data = {
    '密度':[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
    '含糖率':[0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103],
    '好瓜':[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
    }

def get_watermelonDataset():
    return pd.DataFrame(data)

def preprocess_watermelon_data():
    data = get_watermelonDataset()
    target = data['好瓜']
    X = data[['密度','含糖率']]
    return np.array(X),np.array(target)

class loglikeRegression():
    def __init__(self,epoch):
        self.epoch = epoch
    def forward(self):
        pass
    def p1(self,X_hat,beta):
        z = np.matmul(beta.T,X_hat)
        if z>10:
            return 1/1+math.e**(-z)
        else:
            return math.e**z/(1+math.e**z)
    def fit(self,X,y):
        X_hat = np.insert(X,2,np.ones((1,X.shape[0])),axis=1)
        # beta = np.random.rand((X_hat.shape[-1]))
        self.beta = np.zeros((X_hat.shape[-1],1))
        for i in tqdm(range(self.epoch)):
            dbeta = 0
            d2beta = np.zeros((len(self.beta),len(self.beta)))
            for i,x_i in enumerate(X_hat):
                x_i = x_i.reshape(-1,1)
                dbeta += x_i*(y[i] - self.p1(x_i,self.beta))
                d2beta += np.matmul(x_i,x_i.T) * self.p1(x_i,self.beta)*(1-self.p1(x_i,self.beta))
            self.beta = self.beta+np.matmul(np.linalg.inv(d2beta),dbeta)
    def predict(self,x):
        x = np.insert(x,x.shape[-1],np.ones((1,x.shape[0])),axis=1)
        z = np.matmul(self.beta.T,x.T)
        rate = 1/(1+math.e**(-z))
        classify = np.where(rate>0.5,1,0)
        return rate,classify
    def accuracy_info(self,x,y):
        _,y_predict = self.predict(x)
        y_predict = y_predict.squeeze()
        y = y.squeeze()
        TP=FP=TN=FN = 0
        for i in range(len(y_predict)):
            if y[i]==1 and y_predict[i]==1:
                TP += 1
            elif y[i]==0 and y_predict[i]==1:
                FP += 1
            elif y[i]==0 and y_predict[i]==0:
                TN += 1
            else:
                FN += 1
        return TP,FP,TN,FN

