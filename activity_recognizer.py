import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm, model_selection, metrics
from DIPPID import SensorUDP
from utils import DIRECTORY, LABEL_DICT
from collections import deque

PORT = 5700
LIVE_DATA_SIZE = 50

class Activity:

    def __init__(self):
        self.live_data = deque(maxlen=LIVE_DATA_SIZE)
        self.order = 1
        self.sampling_rate = 100
        self.cutoff_frequency = 3
        self.butter_filter = signal.butter(self.order, self.cutoff_frequency, btype="low", analog=False, output="sos", fs=self.sampling_rate)
        self.finished_training = False
        self.got_live_data = False
        self.sensor = SensorUDP(PORT)

        self.classifier = None
        self.scaler = None

    # Filter the signal

    def apply_filter(self, df):
        df = df.copy()
        df["acc_x_filtered"] = signal.sosfilt(self.butter_filter, df["acc_x"])
        df["acc_y_filtered"] = signal.sosfilt(self.butter_filter, df["acc_y"])
        df["acc_z_filtered"] = signal.sosfilt(self.butter_filter, df["acc_z"])
        df["gyro_x_filtered"] = signal.sosfilt(self.butter_filter, df["gyro_x"])
        df["gyro_y_filtered"] = signal.sosfilt(self.butter_filter, df["gyro_y"])
        df["gyro_z_filtered"] = signal.sosfilt(self.butter_filter, df["gyro_z"])
        return df


    # Get the dominant frequency of the signal

    def get_dominant_frequency(self, data):
        spectrum = np.abs(np.fft.fft(data))
        frequencies = np.fft.fftfreq(len(data), 1/self.sampling_rate)
        mask = frequencies >= 0
        spectrum = spectrum[mask]
        frequencies = frequencies[mask]
        return frequencies[np.argmax(spectrum)]


    # Preprocess the training data and get the features used for training

    def extract_features(self, csv_df, label=None):
        csv_df = csv_df.dropna()
        csv_df = self.apply_filter(csv_df)
        features = {
            "max_acc_x": csv_df["acc_x_filtered"].max(),
            "min_acc_x": csv_df["acc_x_filtered"].min(),
            "mean_acc_x": csv_df["acc_x_filtered"].mean(),
            "dominant_freq_acc_x": self.get_dominant_frequency(csv_df["acc_x_filtered"]),
            "std_acc_x": csv_df["acc_x_filtered"].std(),
            "var_acc_x": csv_df["acc_x_filtered"].var(),
            "max_acc_y": csv_df["acc_y_filtered"].max(),
            "min_acc_y": csv_df["acc_y_filtered"].min(),
            "mean_acc_y": csv_df["acc_y_filtered"].mean(),
            "dominant_freq_acc_y": self.get_dominant_frequency(csv_df["acc_y_filtered"]),
            "std_acc_y": csv_df["acc_y_filtered"].std(),
            "var_acc_y": csv_df["acc_y_filtered"].var(),
            "max_acc_z": csv_df["acc_z_filtered"].max(),
            "min_acc_z": csv_df["acc_z_filtered"].min(),
            "mean_acc_z": csv_df["acc_z_filtered"].mean(),
            "dominant_freq_acc_z": self.get_dominant_frequency(csv_df["acc_z_filtered"]),
            "std_acc_z": csv_df["acc_z_filtered"].std(),
            "var_acc_z": csv_df["acc_z_filtered"].var(),
            "max_gyro_x": csv_df["gyro_x_filtered"].max(),
            "min_gyro_x": csv_df["gyro_x_filtered"].min(),
            "mean_gyro_x": csv_df["gyro_x_filtered"].mean(),
            "dominant_freq_gyro_x": self.get_dominant_frequency(csv_df["gyro_x_filtered"]),
            "std_gyro_x": csv_df["gyro_x_filtered"].std(),
            "var_gyro_x": csv_df["gyro_x_filtered"].var(),
            "max_gyro_y": csv_df["gyro_y_filtered"].max(),
            "min_gyro_y": csv_df["gyro_y_filtered"].min(),
            "mean_gyro_y": csv_df["gyro_y_filtered"].mean(),
            "dominant_freq_gyro_y": self.get_dominant_frequency(csv_df["gyro_y_filtered"]),
            "std_gyro_y": csv_df["gyro_y_filtered"].std(),
            "var_gyro_y": csv_df["gyro_y_filtered"].var(),
            "max_gyro_z": csv_df["gyro_z_filtered"].max(),
            "min_gyro_z": csv_df["gyro_z_filtered"].min(),
            "mean_gyro_z": csv_df["gyro_z_filtered"].mean(),
            "dominant_freq_gyro_z": self.get_dominant_frequency(csv_df["gyro_z_filtered"]),
            "std_gyro_z": csv_df["gyro_z_filtered"].std(),
            "var_gyro_z": csv_df["gyro_z_filtered"].var(),
    }

        if label is not None:
            features["label"] = label

        return pd.DataFrame([features])


    # Get the incoming data from the input device

    def get_live_data(self):
        if self.sensor.has_capability('accelerometer') and self.sensor.has_capability('gyroscope'):
            acc_data = self.sensor.get_value('accelerometer')
            gyro_data = self.sensor.get_value('gyroscope')
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
    
    def train_classifier(self):
        print("Starting classifier training...")
        data_list = []
        # Load all csv files in DIRECTORY. Get the label from their subdirectory and extract their features
        for csv in Path(DIRECTORY).rglob('*.csv'):
            csv_df = pd.read_csv(csv)
            label = csv.parent.name
            data_list.append(self.extract_features(csv_df, label=label))

        df = pd.concat(data_list, ignore_index=True)

        df_to_scale = df.drop(columns=['label'])
        df['class'] = df['label'].map(LABEL_DICT)

        scaler = StandardScaler()
        scaled_samples = scaler.fit_transform(df_to_scale)

        features_train, features_test, classes_train, classes_test = model_selection.train_test_split(scaled_samples, df['class'], test_size=0.2)

        classifier = OneVsOneClassifier(svm.SVC(kernel='poly'))
        classifier.fit(features_train, classes_train)

        pred = classifier.predict(features_test)
        print(f"Accuracy: {metrics.accuracy_score(classes_test, pred):.2%}")
        print("Classifier training complete.")
        self.finished_training = True
        return classifier, scaler
    

    def predict_live_data(self):
        new_data = self.get_live_data()
        if new_data:
            self.live_data.append(new_data)

        if len(self.live_data) == LIVE_DATA_SIZE:
            self.got_live_data = True
            feature_df = self.extract_features(pd.DataFrame(self.live_data))
            ordered_df = feature_df[self.scaler.feature_names_in_]
            
            ordered_df_scaled = self.scaler.transform(ordered_df)

            pred_class = self.classifier.predict(ordered_df_scaled)[0]
            pred_label = None
            for key, value in LABEL_DICT.items():
                if value == pred_class:
                    pred_label = key
                    break
                
            return pred_label