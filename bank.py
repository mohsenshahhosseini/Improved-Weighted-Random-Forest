"""
@author: Mohsen
Improved Weighted Random forest - Bank Marketing data set
"""

import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn import metrics
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import time
# import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import os


pd.set_option('display.max_columns', 500)

bank = pd.read_csv('bank-full.csv', header=0, sep=';')
sum(bank.isnull().sum(axis=1))
dummies = pd.get_dummies(bank.drop(columns=['y']))
dummies['class'] = bank['y']
X = dummies.drop(columns='class')
Y = dummies['class']
Y = Y.replace('no', 0)
Y = Y.replace('yes', 1)

resampling = 50
results_total = {}

for x in range(resampling):

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    R2 = RandomForestClassifier(n_estimators=100, max_depth=round(np.sqrt(x_train.shape[1])/2))
    RF2 = R2.fit(x_train,y_train)
    RF_predict = RF2.predict(x_test)
    RF_prob = RF2.predict_proba(x_test)[:,1]
    RF_score = accuracy_score(y_test, RF_predict)
    RF_AUC = roc_auc_score(y_test, RF_prob)

    R2_deep = RandomForestClassifier(n_estimators=100)
    RF2_deep = R2_deep.fit(x_train,y_train)
    RF_deep_predict = RF2_deep.predict(x_test)
    RF_deep_prob = RF2_deep.predict_proba(x_test)[:,1]
    RF_deep_score = accuracy_score(y_test, RF_predict)
    RF_deep_AUC = roc_auc_score(y_test, RF_prob)

    estimators_list = []
    for i in range(R2.n_estimators):
         estimators_list.append(R2.estimators_[i])
    estimators_list2 = estimators_list.copy()
    estimators_list2 = pd.DataFrame(estimators_list2)

    cv_df = []
    cv_df2 = []
    ws = []
    we = []
    # each_df = pd.DataFrame()
    for i in range(R2.n_estimators):
        ES = R2.estimators_[i].fit(x_train, y_train)
        cvv = cross_val_predict(ES, x_train, y_train, cv=5)
        cvv2 = cross_val_predict(ES, x_train, y_train, cv=5, method='predict_proba')[:,1]
        cv_df.append(cvv)
        cv_df2.append(cvv2)
        ws.append(accuracy_score(y_train, cvv))
        we.append(np.exp(accuracy_score(y_train, cvv)))
    sums = sum(ws)
    for j in range(len(ws)):
        ws[j] = ws[j]/sums
    sum_we = sum(we)
    for k in range(len(we)):
        we[k] = we[k]/sum_we
    cv_df = pd.DataFrame(cv_df).T
    cv_df2 = pd.DataFrame(cv_df2).T


    ### Scipy minimize - accuracy
    def objective2(y):
        return accuracy_score(y_train, round(sum(y[i]*cv_df[i] for i in range(R2.n_estimators))))
    def constraint2(y):
        return sum(y[i] for i in range(R2.n_estimators)) - 1.0
    y0 = np.zeros(R2.n_estimators)
    for i in range(R2.n_estimators):
        y0[i] = 1/R2.n_estimators
    b = (0, 1.0)
    bnds = tuple([b]*R2.n_estimators)
    con2 = {'type': 'eq', 'fun': constraint2}
    solution2 = minimize(objective2, y0, method='SLSQP',
                         options={'disp': True, 'maxiter': 3000, 'eps':1e-3}, bounds=bnds,
                         constraints=con2)
    y = solution2.x
    ensemble_preds_test = sum(y[i]*estimators_list[i].predict(x_test) for i in range(R2.n_estimators)).round()
    ensemble_score_test = accuracy_score(y_test, ensemble_preds_test)


    ### Scipy minimize - AUC
    def objective3(z):
        return roc_auc_score(y_train, sum(z[i] * cv_df2[i] for i in range(R2.n_estimators)))
    def constraint3(z):
        return sum(z[i] for i in range(R2.n_estimators)) - 1.0
    z0 = np.zeros(R2.n_estimators)
    for i in range(R2.n_estimators):
        z0[i] = 1/R2.n_estimators
    b = (0, 1.0)
    bnds3 = tuple([b]*R2.n_estimators)
    con3 = {'type': 'eq', 'fun': constraint3}
    solution3 = minimize(objective3, z0, method='SLSQP',
                         options={'disp': True, 'maxiter': 3000, 'eps':1e-3}, bounds=bnds3,
                         constraints=con3)
    z = solution3.x
    ensemble_preds_test3 = sum(z[i]*estimators_list[i].predict(x_test) for i in range(R2.n_estimators)).round()
    ensemble_score_test3 = accuracy_score(y_test, ensemble_preds_test3)


    #### stacking-RF-Accuracy
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    stacked_RF = RandomForestClassifier(n_estimators=100)
    stacked_RF.fit(cv_df, y_train)
    stacked_RF_preds_test = pd.DataFrame(stacked_RF.predict(pd.DataFrame([estimators_list2.iloc[i,:].values[0].predict(x_test) for i in range(cv_df.shape[1])]).T)).round()
    stacked_RF_test_score = accuracy_score(y_test, stacked_RF_preds_test)

    #### stacking-RF-AUC
    stacked_RF_A = RandomForestClassifier(n_estimators=100)
    stacked_RF_A.fit(cv_df2, y_train)
    stacked_RF_A_preds_test = pd.DataFrame(stacked_RF_A.predict(pd.DataFrame([estimators_list2.iloc[i,:].values[0].predict(x_test) for i in range(cv_df2.shape[1])]).T)).round()
    stacked_RF_A_test_score = accuracy_score(y_test, stacked_RF_A_preds_test)

    #### stacking-Log-Accuracy
    stacked_log = LogisticRegression()
    stacked_log.fit(cv_df, y_train)
    stacked_log_preds_test = pd.DataFrame(stacked_log.predict(pd.DataFrame([estimators_list2.iloc[i,:].values[0].predict(x_test) for i in range(cv_df.shape[1])]).T)).round()
    stacked_log_test_score = accuracy_score(y_test, stacked_log_preds_test)

    #### stacking-Log-AUC
    stacked_log_A = LogisticRegression()
    stacked_log_A.fit(cv_df2, y_train)
    stacked_log_A_preds_test = pd.DataFrame(stacked_log_A.predict(pd.DataFrame([estimators_list2.iloc[i,:].values[0].predict(x_test) for i in range(cv_df2.shape[1])]).T)).round()
    stacked_log_A_test_score = accuracy_score(y_test, stacked_log_A_preds_test)

    #### stacking-KNN-Accuracy
    stacked_knn = KNeighborsClassifier()
    stacked_knn.fit(cv_df, y_train)
    stacked_knn_preds_test = pd.DataFrame(stacked_knn.predict(pd.DataFrame([estimators_list2.iloc[i,:].values[0].predict(x_test) for i in range(cv_df.shape[1])]).T)).round()
    stacked_knn_test_score = accuracy_score(y_test, stacked_knn_preds_test)

    #### stacking-KNN-AUC
    stacked_knn_A = KNeighborsClassifier()
    stacked_knn_A.fit(cv_df2, y_train)
    stacked_knn_A_preds_test = pd.DataFrame(stacked_knn_A.predict(pd.DataFrame([estimators_list2.iloc[i,:].values[0].predict(x_test) for i in range(cv_df2.shape[1])]).T)).round()
    stacked_knn_A_test_score = accuracy_score(y_test, stacked_knn_A_preds_test)

    #### ensemble-ws
    ensemblew_preds_test = sum(ws[i]*estimators_list[i].predict(x_test) for i in range(R2.n_estimators)).round()
    ensemblew_score_test =accuracy_score(y_test, ensemblew_preds_test)

    #### ensemble-we
    ensemble_we_preds_test = sum(we[i]*estimators_list[i].predict(x_test) for i in range(R2.n_estimators)).round()
    ensemble_we_score_test = accuracy_score(y_test, ensemble_we_preds_test)


    results_total[x] = pd.DataFrame({'RF - Shallow':[RF_score], 'RF - Deep':[RF_deep_score],
                                     'Optimized RF - Accuracy':[ensemble_score_test], 'Optimized RF - AUC': [ensemble_score_test3],
                                     'Stacked RF - Binary':[stacked_RF_test_score], 'Stacked RF - Prob':[stacked_RF_A_test_score],
                                     'Stacked Log - Binary': [stacked_log_test_score], 'Stacked Log - Prob': [stacked_log_A_test_score],
                                     'Stacked KNN - Binary': [stacked_knn_test_score], 'Stacked KNN - Prob': [stacked_knn_A_test_score],
                                     'Weighted RF':[ensemblew_score_test], 'E-weighted RF':[ensemble_we_score_test]})


results_avg = [sum(results_total[i].iloc[0,:] for i in range(resampling))][0]/resampling
results_avg.to_csv('bank_results.csv')
