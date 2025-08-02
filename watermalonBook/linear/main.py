import untils
import numpy as np
from sklearn.model_selection import train_test_split

X,y  = untils.preprocess_watermelon_data()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

loglinearModel = untils.loglikeRegression(100)
loglinearModel.fit(X_train,y_train)
TP,FP,TN,FN = loglinearModel.accuracy_info(X_test,y_test)

accuracy = (TP+TN)/(TP+FP+TN+FN)
print(accuracy)
