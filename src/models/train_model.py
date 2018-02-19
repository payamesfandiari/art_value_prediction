import pandas as pd
import numpy as np
from matplotlib import  pyplot as plt
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor,
                              ExtraTreesRegressor
                              )
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection

from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from time import time


def tuner(clf, x, y, params, cv_fold=5, n_iter=10):
    cv = model_selection.RandomizedSearchCV(estimator=clf, param_distributions=params, n_iter=n_iter
                                            , scoring={"r2": metrics.make_scorer(metrics.r2_score)
            , "mse": "neg_mean_squared_error"}, cv=cv_fold, verbose=2
                                            , return_train_score=True, refit="r2")
    cv.fit(x, y)
    return cv


def benchmark(clf, X_train, y_train, X_test, y_test, params, feature_names=None, **kwargs):
    print('_' * 80)
    print("Training: ")
    print(str(clf).split('(')[0])
    t0 = time()

    cv_res = tuner(clf, X_train, y_train, params=params, n_iter=kwargs.get('n_iter', 10)
                   , cv_fold=kwargs.get('cv_fold', 5))
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = cv_res.best_estimator_.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print("r2:", r2)
    print("Mean Absolute Error :", mae)
    print("Mean Sq Error :", mse)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])

        if feature_names is not None:
            importance = cv_res.best_estimator_.coef_
            x = np.argsort(importance)[-20:]
            plt.bar(range(0, len(x)), importance[x])
            plt.xticks(range(0, len(x)), tuple(feature_names[x]), rotation='vertical')
            plt.show()
            print("top 10 keywords per class:")
            print(" ".join(feature_names[x]))
        print()
    elif hasattr(clf, 'feature_importances_'):
        if feature_names is not None:
            importance = cv_res.best_estimator_.feature_importances_
            x = np.argsort(importance)[-20:]
            plt.bar(range(0, len(x)), importance[x])
            plt.xticks(range(0, len(x)), tuple(feature_names[x]), rotation='vertical')
            plt.show()
            print("top 10 keywords per class:")
            print(" ".join(feature_names[x]))
        print()

    print()
    return cv_res.best_estimator_, pd.DataFrame(cv_res.cv_results_), mse, mae, r2, train_time, test_time


