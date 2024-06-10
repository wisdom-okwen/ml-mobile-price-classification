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
