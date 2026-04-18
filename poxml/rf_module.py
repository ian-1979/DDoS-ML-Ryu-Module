import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize, LabelBinarizer

from util import print_success, print_err

warnings.filterwarnings('ignore')

# constant: feature names from dataset
# dataset sourced from: https://github.com/neelimabonangi/Ddos-detection-sdn-ml
FEATURE_NAMES = [
        'dt', 'switch', 'src', 'dst', 'pkt_count', 'byte_count', 'duration',
        'duration_nsec', 'total_duration', 'flows', 'pkt_rate', 'pair_flow',
        'protocol', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps',
        'tot_kbps', 'label', 'packetins', 'byte_per_flow', 'packet_per_flow'
    ]

# <editor-fold desc=>

def load_training_data():
    d = pd.read_csv('dataset.csv')
    return d

def process_training_data():
    # load dataset
    dataset = load_training_data()

    #encode protocol column
    dataset['protocol'] = dataset['protocol'].astype('category').cat.codes

    #round values for better generalization
    dataset['pktrate'] = dataset['pktrate'].round(1)
    dataset['tx_kbps'] = dataset['tx_kbps'].round(1)
    dataset['rx_kbps'] = dataset['rx_kbps'].round(1)
    dataset['tot_kbps'] = dataset['tot_kbps'].round(1)

    dataset['dt'] = pd.to_numeric(dataset['dt'], errors='coerce')

    dataset['time_bin'] = dataset['dt'] // 10  # 10-unit window

    dataset_grouped = dataset.groupby('time_bin').agg({
        'pktCount': 'sum',
        'byteCount': 'sum',
        'flows': 'sum',
        'pktrate': 'mean',
        'label': 'max'
    }).reset_index()

    x = dataset_grouped.drop(columns=['label', 'time_bin'])
    y = dataset_grouped['label']

    # splitting data into training and validation vs test data
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    # splitting training vs validation
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


    return x_train, x_val, x_test, y_train, y_val, y_test

def train_model():
    # get training data
    x_train, x_val, x_test, y_train, y_val, y_test = process_training_data()
    # train model
    s_time = time.time() # start timer
    rfc = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        oob_score=True
    )
    print("Training model...")
    rfc.fit(x_train, y_train)
    t_time = time.time() - s_time  # time taken to train

    print(f"Time taken to train model: {t_time}")
    print(f"OOB score: {rfc.oob_score_}")

    importances = pd.DataFrame({
        "feature": x_train.columns,
        "importance": rfc.feature_importances_
    })

    importances = importances.sort_values(by="importance", ascending=False)

    print(importances)

    # weights of each feature on decision making
    fi = rfc.feature_importances_

    # save model for later
    joblib.dump(rfc, 'rfc.pkl')

    results = {
        "model": rfc,
        "x_val": x_val,
        "x_test": x_test,
        "y_val": y_val,
        "y_test": y_test,
        "exe_time": t_time,
        "fi": fi,
    }
    return results

def load_ml():
    x_train, x_val, x_test, y_train, y_val, y_test = process_training_data()
    ml = joblib.load('rfc.pkl')
    t_time = 0
    results = {
        "model": ml,
        "x_val": x_val,
        "x_test": x_test,
        "y_val": y_val,
        "y_test": y_test,
        "exe_time": t_time,
        "fi": ml.feature_importances_,
    }
    print(results)
    return results


def calculate_model_metrics_val(r):
    print("Calculate Model Metrics (Validation)...")
    y_val_pred = r["model"].predict(r["x_val"])
    print(f"y_val_pred: {y_val_pred}")

    # calculate metrics
    val_accuracy = accuracy_score(r["y_val"], y_val_pred)
    print(f"Validation accuracy: {val_accuracy}")
    val_precision = precision_score(r["y_val"], y_val_pred)
    print(f"Validation precision: {val_precision}")
    val_recall = recall_score(r["y_val"], y_val_pred)
    print(f"Validation recall: {val_recall}")
    val_f1 = f1_score(r["y_val"], y_val_pred)
    print(f"Validation f1: {val_f1}")

def calculate_model_metrics_test(r):
    print("Calculate Model Metrics (Test)...")

    y_test_pred = r["model"].predict(r["x_test"])
    # need to convert to binary options since
    #y_test_pred_binary = (y_test_pred >= 0.5).astype(int)
    test_accuracy = accuracy_score(r["y_test"], y_test_pred)
    print(f"test accuracy: {test_accuracy}")
    test_precision = precision_score(r["y_test"], y_test_pred)
    print(f"test precision: {test_precision}")
    test_recall = recall_score(r["y_test"], y_test_pred)
    print(f"test recall: {test_recall}")
    test_f1 = f1_score(r["y_test"], y_test_pred)
    print(f"test F1: {test_f1}")

# <editor-fold desc="TEST FUNCTIONS">

def test_load_training_data():
    t = load_training_data()
    #print(t)
    t.info()
    print_success("test_load_training_data")

def test_process_training_data():
    x1, x2, x3, y1, y2, y3 = process_training_data()
    print(x1)
    print(x2)
    print(x3)
    print(y1)
    print(y2)
    print(y3)
    print_success("test_process_training_data")

def test_train_model():
    return train_model()

def test_load_ml():
    return load_ml()


def test_calculate_model_metrics(results):

    calculate_model_metrics_val(results)
    print_success("test_calculate_model_metrics_val")
    calculate_model_metrics_test(results)
    print_success("test_calculate_model_metrics_test")
    print_success("test_calculate_model_metrics")

def test_iloc():
    data = load_training_data()

    x = data.drop(data.columns[-4], axis=1)
    y = data.iloc[:, -4]
    print(x)
    print(y)

# used for testing
if __name__ == "__main__":
    # todo:
    # create graph of test predictions


    # Tests
    #test_iloc()
    test_load_training_data()
    test_process_training_data()
    x = test_train_model()
    #x = test_load_ml()
    #print_success("test_load_ml")
    test_calculate_model_metrics(x)
    print_success("rf_module tests")

# </editor-fold>


