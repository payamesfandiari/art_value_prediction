import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              )
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from mlxtend.regressor import StackingCVRegressor

from time import time
from sklearn.externals import joblib


def rmse(ground_truth, predictions):
    mse = metrics.mean_squared_error(ground_truth, predictions)
    return np.sqrt(mse)


def tuner(clf, x, y, params, cv_fold=5, n_iter=10):
    """
    Do a randomized search cross validation on the given estimator
    :param clf: A scikit-learn type Estimator which has .fit() and .predict() methods
    :param x: A numpy array of shape n x m with n instances and m features
    :param y: A vector of n values
    :param params: The set of parameters for the Estimator to be looked at and searched
    :param cv_fold: number of cross validation folds
    :param n_iter: number of iterations to run RandomizedSearchCV
    :return:
    """
    cv = model_selection.RandomizedSearchCV(estimator=clf, param_distributions=params, n_iter=n_iter
                                            , scoring={"r2": metrics.make_scorer(metrics.r2_score)
            , "rmse": metrics.make_scorer(rmse, greater_is_better=False)}, cv=cv_fold, verbose=2
                                            , return_train_score=True, refit="r2")
    cv.fit(x, y)
    return cv


def benchmark(clf, X_train, y_train, X_test, y_test, params, feature_names=None, **kwargs):
    """
    Runs some benchmarking on the given Regression algorithm
    :param clf: A scikit-learn type Estimator which has .fit() and .predict() methods
    :param X_train: A numpy array of shape n x m with n instances and m features
    :param y_train: A vector of n values
    :param X_test: A test set
    :param y_test: Regression targets for the test set
    :param params: The set of parameters for the Estimator to be looked at and searched
    :param feature_names: Name of features.
    :param kwargs: n_iter : number of iterations for Randomized Search. cv_fold : number of cross validation folds
    :return:
    """
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
            print("top 10 keywords per class:")
            print(" ".join(feature_names[x]))
        print()
    elif hasattr(clf, 'feature_importances_'):
        if feature_names is not None:
            importance = cv_res.best_estimator_.feature_importances_
            x = np.argsort(importance)[-20:]
            print("top 10 keywords per class:")
            print(" ".join(feature_names[x]))
        print()

    print()
    return cv_res.best_estimator_, pd.DataFrame(cv_res.cv_results_), mse, mae, r2, train_time, test_time


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    data = pd.read_csv("../../data/processed/data.csv", index_col=0, encoding='latin-1')
    pred = list(data.columns)
    pred.remove("hammer_price")
    lbl = data["hammer_price"]
    data = data[pred]
    train, test, y_train, y_test = model_selection.train_test_split(data, lbl, train_size=0.7, test_size=0.3)
    # Finding the best set of Parameters for the level1 regressors.
    param_grid = {
        'n_estimators': [100, 50, 500, 1000],
        'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.7],
        'loss': ['linear', 'square', 'exponential'],
        'base_estimator__splitter': ['best', 'random'],
        'base_estimator__max_features': [3, 5, 'sqrt', 0.3],
    }
    clf = AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_features=3))
    res = benchmark(clf, train.values, y_train.values, test.values, y_test.values, param_grid,
                    np.array(train.columns), n_iter=10, cv_fold=3)
    print("Best parameters are : ",
          res[1].sort_values('mean_test_rmse', ascending=False).iloc[[0, 1, 2, 3], "params"].values)

    # Identifying the best meta-regressors's params
    rf = RandomForestRegressor(n_jobs=-1, verbose=1)
    stack = StackingCVRegressor(
        regressors=(
            AdaBoostRegressor(n_estimators=50, loss='square', learning_rate=0.3,
                              base_estimator=tree.DecisionTreeRegressor(max_features=0.3, splitter='best')),
            AdaBoostRegressor(n_estimators=50, loss='square', learning_rate=0.3,
                              base_estimator=tree.DecisionTreeRegressor(max_features=0.5, splitter='best')),
            AdaBoostRegressor(n_estimators=50, loss='square', learning_rate=0.5,
                              base_estimator=tree.DecisionTreeRegressor(max_features=3)),
            AdaBoostRegressor(n_estimators=500, loss='exponential', learning_rate=0.1,
                              base_estimator=tree.DecisionTreeRegressor(max_features=3, splitter='random')),
        ), meta_regressor=rf, use_features_in_secondary=True)

    param_grid = {
        'meta-randomforestregressor__n_estimators': [10, 50, 100, 500],
        'meta-randomforestregressor__max_features': [3, 5, 0.3, 0.5, 0.7]
    }

    res = benchmark(stack, train.values, y_train.values, test.values, y_test.values, param_grid,
                    np.array(train.columns), n_iter=10, cv_fold=3)

    print("Best parameter are : ",
          res[1].sort_values('mean_test_rmse', ascending=False).iloc[[0], "params"].values)

    print("Training the final model to be used...")
    # Training the final model and saving it in ../../models for future use.
    rf_final = RandomForestRegressor(n_jobs=-1, verbose=1, max_features=3, n_estimators=50)
    stack_final = StackingCVRegressor(
        regressors=(
            AdaBoostRegressor(n_estimators=50, loss='square', learning_rate=0.3,
                              base_estimator=tree.DecisionTreeRegressor(max_features=0.3, splitter='best')),
            AdaBoostRegressor(n_estimators=50, loss='square', learning_rate=0.3,
                              base_estimator=tree.DecisionTreeRegressor(max_features=0.5, splitter='best')),
            AdaBoostRegressor(n_estimators=50, loss='square', learning_rate=0.5,
                              base_estimator=tree.DecisionTreeRegressor(max_features=3)),
            AdaBoostRegressor(n_estimators=500, loss='exponential', learning_rate=0.1,
                              base_estimator=tree.DecisionTreeRegressor(max_features=3, splitter='random')),
        ), meta_regressor=rf_final, use_features_in_secondary=True, store_train_meta_features=False)
    stack_final.fit(data.values, lbl.values)
    joblib.dump(stack_final, '../../models/stacked_regressor.pkl',compress=3)


if __name__ == '__main__':
    main()
