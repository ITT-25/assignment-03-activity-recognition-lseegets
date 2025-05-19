# this program recognizes activities
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from sklearn import svm # scikit-learn
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import svm, model_selection, metrics
from DIPPID import SensorUDP
from utils import DIRECTORY, LABEL_DICT
from collections import deque
import time

PORT = 5700
LIVE_DATA_SIZE = 50

sensor = SensorUDP(PORT)
live_data = deque(maxlen=LIVE_DATA_SIZE)
order = 1
sampling_rate = 100
cutoff_frequency = 3
butter_filter = signal.butter(order, cutoff_frequency, btype="low", analog=False, output="sos", fs=sampling_rate)


# Filter the signal

def apply_filter(df):
    df["acc_x_filtered"] = signal.sosfilt(butter_filter, df["acc_x"])
    df["acc_y_filtered"] = signal.sosfilt(butter_filter, df["acc_y"])
    df["acc_z_filtered"] = signal.sosfilt(butter_filter, df["acc_z"])
    df["gyro_x_filtered"] = signal.sosfilt(butter_filter, df["gyro_x"])
    df["gyro_y_filtered"] = signal.sosfilt(butter_filter, df["gyro_y"])
    df["gyro_z_filtered"] = signal.sosfilt(butter_filter, df["gyro_z"])
    return df


# Get the dominant frequency of the signal

def get_dominant_frequency(data):
    spectrum = np.abs(np.fft.fft(data))
    frequencies = np.fft.fftfreq(len(data), 1/sampling_rate)
    mask = frequencies >= 0
    spectrum = spectrum[mask]
    frequencies = frequencies[mask]
    return frequencies[np.argmax(spectrum)]


# Preprocess the training data and get the features used for training

def extract_features(csv_df, label=None):
    csv_df = csv_df.dropna()
    csv_df = apply_filter(csv_df)
    features = {
        "max_acc_x": csv_df["acc_x_filtered"].max(),
        "min_acc_x": csv_df["acc_x_filtered"].min(),
        "mean_acc_x": csv_df["acc_x_filtered"].mean(),
        "dominant_freq_acc_x": get_dominant_frequency(csv_df["acc_x_filtered"]),
        "max_acc_y": csv_df["acc_y_filtered"].max(),
        "min_acc_y": csv_df["acc_y_filtered"].min(),
        "mean_acc_y": csv_df["acc_y_filtered"].mean(),
        "dominant_freq_acc_y": get_dominant_frequency(csv_df["acc_y_filtered"]),
        "max_acc_z": csv_df["acc_z_filtered"].max(),
        "min_acc_z": csv_df["acc_z_filtered"].min(),
        "mean_acc_z": csv_df["acc_z_filtered"].mean(),
        "dominant_freq_acc_z": get_dominant_frequency(csv_df["acc_z_filtered"]),
        "max_gyro_x": csv_df["gyro_x_filtered"].max(),
        "min_gyro_x": csv_df["gyro_x_filtered"].min(),
        "mean_gyro_x": csv_df["gyro_x_filtered"].mean(),
        "dominant_freq_gyro_x": get_dominant_frequency(csv_df["gyro_x_filtered"]),
        "max_gyro_y": csv_df["gyro_y_filtered"].max(),
        "min_gyro_y": csv_df["gyro_y_filtered"].min(),
        "mean_gyro_y": csv_df["gyro_y_filtered"].mean(),
        "dominant_freq_gyro_y": get_dominant_frequency(csv_df["gyro_y_filtered"]),
        "max_gyro_z": csv_df["gyro_z_filtered"].max(),
        "min_gyro_z": csv_df["gyro_z_filtered"].min(),
        "mean_gyro_z": csv_df["gyro_z_filtered"].mean(),
        "dominant_freq_gyro_z": get_dominant_frequency(csv_df["gyro_z_filtered"]),
    }

    if label is not None:
        features["label"] = label

    return pd.DataFrame([features])


# Get the incoming data from the input device

def get_live_data():
    if sensor.has_capability('accelerometer') and sensor.has_capability('gyroscope'):
        acc_data = sensor.get_value('accelerometer')
        gyro_data = sensor.get_value('gyroscope')
        return {
            'acc_x': acc_data['x'],
            'acc_y': acc_data['y'],
            'acc_z': acc_data['z'],
            'gyro_x': gyro_data['x'],
            'gyro_y': gyro_data['y'],
            'gyro_z': gyro_data['z']
        }
    return None


# Training the classifier
    
def train_classifier():
    data_list = []
    # Load all csv files in DIRECTORY. Get the label from their subdirectory and extract their features
    for csv in Path(DIRECTORY).rglob('*.csv'):
        csv_df = pd.read_csv(csv)
        label = csv.parent.name
        data_list.append(extract_features(csv_df, label=label))

    df = pd.concat(data_list, ignore_index=True)

    scaled_samples = scale(df.drop(columns=['label']))
    df_mean = df.copy()
    df_mean.loc[:, df.columns != 'label'] = scaled_samples

    scaler = MinMaxScaler()
    scaler.fit(df_mean.loc[:, df_mean.columns != 'label'])
    scaled_samples = scaler.transform(df_mean.loc[:, df_mean.columns != 'label'])
    df_normalized = df_mean.copy()
    df_normalized.loc[:, df_mean.columns != 'label'] = scaled_samples

    df_normalized["class"] = df_normalized["label"].map(LABEL_DICT)

    features = df_normalized.drop(columns=["label", "class"])
    classes = df_normalized["class"]

    features_train, features_test, classes_train, classes_test = model_selection.train_test_split(features, classes, test_size=0.2)

    classifier = OneVsOneClassifier(svm.SVC(kernel='poly'))
    #classifier = OneVsRestClassifier(svm.SVC(kernel='poly'))
    classifier.fit(features_train, classes_train)

    classes_predicted = classifier.predict(features_test)

    accuracy = metrics.accuracy_score(classes_test, classes_predicted)
    print(f"Accuracy: {accuracy}")

    return classifier


trained_classifier = train_classifier()

while True:
    new_data = get_live_data()
    if new_data:
        live_data.append(new_data)

    if len(live_data) == LIVE_DATA_SIZE:
        feature_df = extract_features(pd.DataFrame(live_data))

        pred_class = trained_classifier.predict(feature_df)[0]
        pred_label = None
        for key, value in LABEL_DICT.items():
            if value == pred_class:
                pred_label = key
                break
        print(f"Predicted activity: {pred_label}")

    time.sleep(0.1)