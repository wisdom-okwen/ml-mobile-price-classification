import sagemaker
import boto3

import joblib
import pathlib
from io import StringIO
import argparse
import os

import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split

def model_fxn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__ == '__main__':
    print('[INFO] Extracting arguments')
    parser = argparse.ArgumentParser()

    """
        Hyperparameters sent by client are passed as command lin arguments to the script
    """
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default-os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default-os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default-os.environ.get("x_train.csv"))
    parser.add_argument("--test-file", type=str, default-os.environ.get("x_test.csv"))

    args, _ = parser.parse_known_args()

    print("SKLearn Version:  ", sklearn.__version__)
    print("Joblib Version:  ", joblib.__version__)

    print("[INFO] Reading Data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    labels = features.pop()

    print("Building training and testing datasets")
    print()

    X_train = train_df[features]
    y_train = train_df[lables]
    X_test = test_df[features]
    y_test = test_df[labels]

    print('Column order:  ')
    print(features)
    print()

    print('Label Columns:  ', lables)
    print()

    print('Data shape:  ')
    print()
    print('---- Shape of Training Data (80%) ----')
    print(X_train.shape)
    print(y_train.shape)

    print()
    print('---- Shape of Testing Data (20%) ----')
    print(X_test.shape)
    print(y_test.shape)
    print()

    print('Training Random Forest Model')
    print()

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print('Model persisted at ', model_path)
    print()

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)

    print()
    print('---- Metrics Results For Testing Data ----')
    print()
    
    print('Total Rows:  ', X_test.shape[0])
    print('[TESTING] Model Accuracy is:  ', test_accuracy)
    print('[TESTING] Testing Report:  ')
    print(test_report)

