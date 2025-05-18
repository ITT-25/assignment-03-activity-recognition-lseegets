from DIPPID import SensorUDP
import pandas as pd
import time
from utils import FILE_PATH

PORT = 5700
HEADERS = ['id', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
RECORDING_SPAN = 10

sensor = SensorUDP(PORT)
start_time = 0
is_recording = False

id = 0

df = pd.DataFrame(columns=HEADERS)
df.to_csv(FILE_PATH, index=False)
print("Ready! Press button_1 to start recording")


def handle_btn(data):
    global is_recording, start_time
    if int(data) == 1:
        start_time = time.time()
        is_recording = True
        print("Is recording: ", is_recording)

sensor.register_callback('button_1', handle_btn)


while True:
    if is_recording and (time.time() - start_time <= RECORDING_SPAN):
        #if sensor.has_capability('accelerometer'):
        acc_df = pd.DataFrame({
            'acc_x': [sensor.get_value('accelerometer')['x']],
            'acc_y': [sensor.get_value('accelerometer')['y']],
            'acc_z': [sensor.get_value('accelerometer')['z']]
        })

        #if sensor.has_capability('gyroscope'):
        gyro_df = pd.DataFrame({
            'gyro_x': [sensor.get_value('gyroscope')['x']],
            'gyro_y': [sensor.get_value('gyroscope')['y']],
            'gyro_z': [sensor.get_value('gyroscope')['z']]
        })

        gen_df = pd.DataFrame({'id': [id], 'timestamp': [time.time()]})
        final_df = pd.concat([gen_df, acc_df, gyro_df], axis=1)
        final_df.to_csv(FILE_PATH, mode='a', index=False, header=False)
        id += 1

    if is_recording and (time.time() - start_time > RECORDING_SPAN):
        is_recording = False
        print("Done recording")
        
    time.sleep(0.0001)