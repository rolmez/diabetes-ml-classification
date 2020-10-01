import pandas as pd
import numpy as np

from sklearn import linear_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

data = pd.read_csv("model2.csv")


#merge
x = data.iloc[:,0:7].values
y = data.iloc[:,7:].values


#train&split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)


scores = list() #accuarcy scores


#Logistic Regression
from sklearn.linear_model import LogisticRegression

params = {'C': np.logspace(-4, 4, 20),
         'penalty': ['l1', 'l2'],
         'solver': ['lbfgs', 'liblinear']}
gs_lr = GridSearchCV(LogisticRegression(random_state=42), params, verbose=1, cv=10)
grid_search = gs_lr.fit(X_train, y_train)
lr = LogisticRegression(C=grid_search.best_params_['C'],
                                    penalty=grid_search.best_params_['penalty'],
                                    solver=grid_search.best_params_['solver'])
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred.round())
print("Logistic Regression Acc: ", acc_lr)
scores.append(acc_lr)


#k Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

params = {'n_neighbors': np.arange(1,50),
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'minkowski']} 
gs_knn = GridSearchCV(KNeighborsClassifier(),
                            params,
                            cv = 4,
                            n_jobs = -1)
grid_search = gs_knn.fit(X_train, y_train)

knn = KNeighborsClassifier(metric=grid_search.best_params_['metric'],
                                        n_neighbors=grid_search.best_params_['n_neighbors'],
                                        weights=grid_search.best_params_['weights'])
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred.round())
print("KNN acc: ", acc_knn)
scores.append(acc_knn)


#SVC
from sklearn.svm import SVC

params = [{'C': [1, 2, 3, 4, 5], 'kernel': ['linear'], 'gamma': ['scale', 'auto']},
        {'C': [1, 2, 3, 4, 5], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']},
        {'C': [1, 2, 3, 4, 5], 'kernel': ['sigmoid'], 'gamma': ['scale', 'auto']}]

classifier = SVC(random_state = 0)
gs = GridSearchCV(estimator = classifier,
                    param_grid = params,
                    cv = 10,
                    n_jobs = -1)

grid_search = gs.fit(X_train, y_train)

svc = SVC(C=grid_search.best_params_["C"],
                kernel=grid_search.best_params_['kernel'],
                gamma=grid_search.best_params_['gamma'])

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = accuracy_score(y_test, y_pred.round())
print("SVC Acc: ", acc_svc)
scores.append(acc_svc)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

params = {'criterion': ['gini', 'entropy'],
            'max_leaf_nodes': list(range(2, 100)),
            'min_samples_split': [2, 3, 4]}
gs_dtc = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)


grid_search = gs_dtc.fit(X_train, y_train)

dtc = DecisionTreeClassifier(criterion=grid_search.best_params_['criterion'],
                                    max_leaf_nodes=grid_search.best_params_['max_leaf_nodes'],
                                    min_samples_split=grid_search.best_params_['min_samples_split'])
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred.round())
#   print("DT acc: ", acc_dt)
scores.append(acc_dt)


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred.round())
print("NB Acc: ", acc_nb)
scores.append(acc_nb)


#Random Forest
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators': [30, 40, 50, 60, 70],
          'max_depth': [6, 8, 10, 12, 14],
         'min_samples_split': [9, 11, 13, 15, 17]}
gs_random_forest = GridSearchCV(RandomForestClassifier(random_state=42), params, verbose=1, cv=10)
grid_search = gs_random_forest.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                max_depth=grid_search.best_params_['max_depth'],
                                min_samples_split=grid_search.best_params_['min_samples_split'],
                                random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred.round())
print(acc_rf)
scores.append(acc_rf)



models = ["LogisticRegressionModel", "KnnModel",
            "SvcModel", "DecisionTreeModel",
            "NaiveBayesModel", "RandomForestModel"]
result = zip(models, scores)
result_set = set(result)
print(result_set)

# from sklearn.metrics import classification_report
# target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
# print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names))


# Joblib

import sklearn.externals
import joblib

joblib.dump(lr, "LogisticRegression.pkl")
joblib.dump(knn, "KNN.pkl")
joblib.dump(svc, "SVC.pkl")
joblib.dump(dtc, "DecisionTree.pkl")
joblib.dump(nb, "GaussianNB.pkl")
joblib.dump(rf, "RandomForest.pkl")
