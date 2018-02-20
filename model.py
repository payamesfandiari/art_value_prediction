"""
This is the final deliverable for the Arthena Data Science Challenge.
"""
import pandas as pd
import numpy as np
from os import path
from glob import iglob
from sklearn.externals import joblib
from sklearn import metrics


def rmse(ground_truth, predictions):
    mse = metrics.mean_squared_error(ground_truth,predictions)
    return np.sqrt(mse)


def predict(input_csv_file):
    """
    The entry point of our experiments. We assume that the training is already done on the Train data.
    We also assume that the features have been transformed correctly
    :param input_csv_file: This is the CSV Test data. It can be an address to an existing file or a Pandas Dataframe
    :return:
    """
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    test = input_csv_file
    print("Reading the input data...")
    if isinstance(input_csv_file,str):
        if not (path.exists(input_csv_file) and path.isfile(input_csv_file)):
            raise FileNotFoundError("The provided path is not valid.")

        test = pd.read_csv(input_csv_file,encoding='latin-1')

    for f in iglob("./models/*"):
        if 'transformer' in f:
            feature_transformer_file_name = f
            break
    else:
        feature_transformer_file_name = "transformer.pkl"
    feature_transformer = joblib.load(feature_transformer_file_name)
    X_t = feature_transformer.transform(test)
    y_test = X_t["hammer_price"]
    y_test = y_test.fillna(0)
    X_t = X_t.drop("hammer_price",axis=1)
    model_name = "stacked_regressor.pkl"
    for f in iglob("./models/*"):
        if 'stacked_regressor' in f:
            model_name = f
            break
    learned_model = joblib.load(model_name)
    rmse_score = rmse(y_test.values,learned_model.predict(X_t.values))
    print("The RMSE score is : ",rmse_score)





