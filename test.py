import pandas as pd
import numpy as np

import sklearn.externals
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    
    #Models
    gnb = joblib.load('GaussianNB.pkl')
    lr = joblib.load('LogisticRegression.pkl')
    knn = joblib.load('KNN.pkl')
    dt = joblib.load('DecisionTree.pkl')
    svc = joblib.load('SVC.pkl')
    rf = joblib.load('RandomForest.pkl')
    
    test = np.array([[35, 28, 74, 0, 0, 0, 0]])
    gnb_pred = gnb.predict(test)
    lr_pred = lr.predict(test)
    knn_pred = knn.predict(test)
    dt_pred = dt.predict(test)
    svc_pred = svc.predict(test)
    rf_pred = rf.predict(test)
    dic = dict({'Gaussian:': gnb_pred,
                'Lr:': lr_pred,
                'KNN::': knn_pred,
                'Dt:': dt_pred,
                'SVC:': svc_pred,
                'RF:': rf_pred})
    print(dic)


if __name__ == "__main__":
    main()