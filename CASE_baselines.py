#!/usr/bin/env python
# coding: utf-8

"""
This script contains functions for running baseline models (Logistic Regression, XGBoost, MLP)
on the CASE dataset for valence and arousal classification.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model

from supervised_models import logit, xgb_model, mlp
from utils import perf_metric
from hexr_self import hexr_self

# --- Data Preprocessing Functions ---
def one_hot_encode(df, class_label):
    """One-hot encode the class column and drop unnecessary columns."""
    ohe = OneHotEncoder()
    df_ohe = pd.DataFrame(ohe.fit_transform(df[[class_label]]).toarray())
    df = df.join(df_ohe)
    for col in ['class1', 'class2', 'Unnamed: 0', 'valence', 'arousal']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    X = df.loc[:, :'emg_trap']
    y = df.iloc[:, 8:]
    return X, y

def split_data_without_normalizing(X, y, label_data_rate, test_size=0.2):
    """Split data into labeled, unlabeled, and test sets without normalization."""
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)
    # x_train = x_train.iloc[:, :].values
    # y_train = y_train.iloc[:, :].values
    # x_test = x_test.iloc[:,:].values
    # y_test = y_test.iloc[:,:].values
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    idx = np.random.permutation(len(y_train))
    label_idx = idx[:int(len(idx)*label_data_rate)]
    unlab_idx = idx[int(len(idx)*label_data_rate):]
    x_unlab = x_train[unlab_idx, :]
    x_train = x_train[label_idx, :]
    y_train = y_train[label_idx, :]
    return x_unlab, x_train, x_test, y_train, y_test

def split_data_with_normalization(X, y, label_data_rate, test_size=0.2):
    """Split data into labeled, unlabeled, and test sets with normalization."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = ['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']
    X[features] = scaler.fit_transform(X[features])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    idx = np.random.permutation(len(y_train))
    label_idx = idx[:int(len(idx)*label_data_rate)]
    unlab_idx = idx[int(len(idx)*label_data_rate):]
    x_unlab = x_train[unlab_idx, :]
    x_train = x_train[label_idx, :]
    y_train = y_train[label_idx, :]
    return x_unlab, x_train, x_test, y_train, y_test

# --- Baseline Model Functions ---
def run_logistic_regression(X, Y, label_data_rate, test_size, metric1, metric2, normalize=False):
    if normalize:
        x_unlab, x_train, x_test, y_train, y_test = split_data_with_normalization(X, Y, label_data_rate, test_size)
    else:
        x_unlab, x_train, x_test, y_train, y_test = split_data_without_normalizing(X, Y, label_data_rate, test_size)
    y_test_hat = logit(x_train, y_train, x_test)
    acc = perf_metric(metric1, y_test, y_test_hat)
    auc = perf_metric(metric2, y_test, y_test_hat)
    return acc, auc

def run_xgboost(X, Y, label_data_rate, test_size, metric1, metric2, normalize=True, n_runs=5):
    accs, aucs = [], []
    for _ in range(n_runs):
        if normalize:
            x_unlab, x_train, x_test, y_train, y_test = split_data_with_normalization(X.copy(), Y, label_data_rate, test_size)
        else:
            x_unlab, x_train, x_test, y_train, y_test = split_data_without_normalizing(X.copy(), Y, label_data_rate, test_size)
        y_test_hat = xgb_model(x_train, y_train, x_test)
        accs.append(perf_metric(metric1, y_test, y_test_hat))
        aucs.append(perf_metric(metric2, y_test, y_test_hat))
    return accs, aucs

def run_mlp(X, Y, label_data_rate, test_size, metric1, metric2, mlp_parameters, normalize=False, n_runs=5):
    if normalize:
        x_unlab, x_train, x_test, y_train, y_test = split_data_with_normalization(X, Y, label_data_rate, test_size)
    else:
        x_unlab, x_train, x_test, y_train, y_test = split_data_without_normalizing(X, Y, label_data_rate, test_size)
    accs, aucs = [], []
    for _ in range(n_runs):
        y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
        accs.append(perf_metric(metric1, y_test, y_test_hat))
        aucs.append(perf_metric(metric2, y_test, y_test_hat))
    return accs, aucs

# --- Main Orchestration Function ---
def main():
    # Experimental parameters
    label_data_rate = 0.1
    metric1 = 'acc'
    metric2 = 'auc'
    test_size = 0.15
    mlp_parameters = dict(hidden_dim=100, epochs=100, activation='relu', batch_size=128, num_layers=2)
    data_path = os.path.join('..', 'Data', 'CASE_2class.csv')

    # Logistic Regression - Valence
    df = pd.read_csv(data_path)
    X, Y = one_hot_encode(df.copy(), 'class1')
    log_acc, log_auc = run_logistic_regression(X, Y, label_data_rate, test_size, metric1, metric2, normalize=False)
    print(f"Logistic Regression Valence Accuracy (no norm): {log_acc}")
    print(f"Logistic Regression Valence AUC (no norm): {log_auc}")
    log_acc, log_auc = run_logistic_regression(X, Y, label_data_rate, test_size, metric1, metric2, normalize=True)
    print(f"Logistic Regression Valence Accuracy (norm): {log_acc}")
    print(f"Logistic Regression Valence AUC (norm): {log_auc}")

    # Logistic Regression - Arousal
    df = pd.read_csv(data_path)
    X, Y = one_hot_encode(df.copy(), 'class2')
    log_acc, log_auc = run_logistic_regression(X, Y, label_data_rate, test_size, metric1, metric2, normalize=False)
    print(f"Logistic Regression Arousal Accuracy (no norm): {log_acc}")
    print(f"Logistic Regression Arousal AUC (no norm): {log_auc}")
    log_acc, log_auc = run_logistic_regression(X, Y, label_data_rate, test_size, metric1, metric2, normalize=True)
    print(f"Logistic Regression Arousal Accuracy (norm): {log_acc}")
    print(f"Logistic Regression Arousal AUC (norm): {log_auc}")

    # XGBoost - Valence
    df = pd.read_csv(data_path)
    X, Y = one_hot_encode(df.copy(), 'class1')
    xgb_accs, xgb_aucs = run_xgboost(X, Y, label_data_rate, test_size, metric1, metric2, normalize=False, n_runs=5)
    print(f"XGBoost Valence Accuracies (no norm): {np.round(xgb_accs, 4)}")
    print(f"XGBoost Valence AUCs (no norm): {np.round(xgb_aucs, 4)}")
    print(f"Average Valence Accuracy (XGBoost, no norm): {round(np.mean(xgb_accs), 4)}")
    print(f"Average Valence AUC (XGBoost, no norm): {round(np.mean(xgb_aucs), 4)}")
    print(f"Std Dev Valence Accuracies (XGBoost, no norm): {round(np.std(xgb_accs), 4)}")
    xgb_accs, xgb_aucs = run_xgboost(X, Y, label_data_rate, test_size, metric1, metric2, normalize=True, n_runs=5)
    print(f"XGBoost Valence Accuracies (norm): {np.round(xgb_accs, 4)}")
    print(f"XGBoost Valence AUCs (norm): {np.round(xgb_aucs, 4)}")
    print(f"Average Valence Accuracy (XGBoost, norm): {round(np.mean(xgb_accs), 4)}")
    print(f"Average Valence AUC (XGBoost, norm): {round(np.mean(xgb_aucs), 4)}")
    print(f"Std Dev Valence Accuracies (XGBoost, norm): {round(np.std(xgb_accs), 4)}")

    # XGBoost - Arousal
    df = pd.read_csv(data_path)
    X, Y = one_hot_encode(df.copy(), 'class2')
    xgb_accs, xgb_aucs = run_xgboost(X, Y, label_data_rate, test_size, metric1, metric2, normalize=False, n_runs=5)
    print(f"XGBoost Arousal Accuracies (no norm): {np.round(xgb_accs, 4)}")
    print(f"XGBoost Arousal AUCs (no norm): {np.round(xgb_aucs, 4)}")
    print(f"Average Arousal Accuracy (XGBoost, no norm): {round(np.mean(xgb_accs), 4)}")
    print(f"Average Arousal AUC (XGBoost, no norm): {round(np.mean(xgb_aucs), 4)}")
    print(f"Std Dev Arousal Accuracies (XGBoost, no norm): {round(np.std(xgb_accs), 4)}")
    xgb_accs, xgb_aucs = run_xgboost(X, Y, label_data_rate, test_size, metric1, metric2, normalize=True, n_runs=5)
    print(f"XGBoost Arousal Accuracies (norm): {np.round(xgb_accs, 4)}")
    print(f"XGBoost Arousal AUCs (norm): {np.round(xgb_aucs, 4)}")
    print(f"Average Arousal Accuracy (XGBoost, norm): {round(np.mean(xgb_accs), 4)}")
    print(f"Average Arousal AUC (XGBoost, norm): {round(np.mean(xgb_aucs), 4)}")
    print(f"Std Dev Arousal Accuracies (XGBoost, norm): {round(np.std(xgb_accs), 4)}")

    # MLP - Valence
    df = pd.read_csv(data_path)
    X, Y = one_hot_encode(df.copy(), 'class1')
    mlp_accs, mlp_aucs = run_mlp(X, Y, label_data_rate, test_size, metric1, metric2, mlp_parameters, normalize=False, n_runs=5)
    print(f"MLP Valence Accuracies (no norm): {np.round(mlp_accs, 4)}")
    print(f"MLP Valence AUCs (no norm): {np.round(mlp_aucs, 4)}")
    print(f"Mean Valence Accuracy (MLP, no norm): {round(np.mean(mlp_accs), 4)}")
    print(f"Mean Valence AUC (MLP, no norm): {round(np.mean(mlp_aucs), 4)}")
    print(f"Std Dev Valence Accuracies (MLP, no norm): {round(np.std(mlp_accs), 4)}")
    mlp_accs, mlp_aucs = run_mlp(X, Y, label_data_rate, test_size, metric1, metric2, mlp_parameters, normalize=True, n_runs=5)
    print(f"MLP Valence Accuracies (norm): {np.round(mlp_accs, 4)}")
    print(f"MLP Valence AUCs (norm): {np.round(mlp_aucs, 4)}")
    print(f"Mean Valence Accuracy (MLP, norm): {round(np.mean(mlp_accs), 4)}")
    print(f"Mean Valence AUC (MLP, norm): {round(np.mean(mlp_aucs), 4)}")
    print(f"Std Dev Valence Accuracies (MLP, norm): {round(np.std(mlp_accs), 4)}")

    # MLP - Arousal
    df = pd.read_csv(data_path)
    X, Y = one_hot_encode(df.copy(), 'class2')
    mlp_accs, mlp_aucs = run_mlp(X, Y, label_data_rate, test_size, metric1, metric2, mlp_parameters, normalize=False, n_runs=5)
    print(f"MLP Arousal Accuracies (no norm): {np.round(mlp_accs, 4)}")
    print(f"MLP Arousal AUCs (no norm): {np.round(mlp_aucs, 4)}")
    print(f"Mean Arousal Accuracy (MLP, no norm): {round(np.mean(mlp_accs), 4)}")
    print(f"Mean Arousal AUC (MLP, no norm): {round(np.mean(mlp_aucs), 4)}")
    print(f"Std Dev Arousal Accuracies (MLP, no norm): {round(np.std(mlp_accs), 4)}")
    mlp_accs, mlp_aucs = run_mlp(X, Y, label_data_rate, test_size, metric1, metric2, mlp_parameters, normalize=True, n_runs=5)
    print(f"MLP Arousal Accuracies (norm): {np.round(mlp_accs, 4)}")
    print(f"MLP Arousal AUCs (norm): {np.round(mlp_aucs, 4)}")
    print(f"Mean Arousal Accuracy (MLP, norm): {round(np.mean(mlp_accs), 4)}")
    print(f"Mean Arousal AUC (MLP, norm): {round(np.mean(mlp_aucs), 4)}")
    print(f"Std Dev Arousal Accuracies (MLP, norm): {round(np.std(mlp_accs), 4)}")

if __name__ == "__main__":
    main()
