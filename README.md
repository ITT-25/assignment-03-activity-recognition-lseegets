[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/EppqwQTz)

### Make sure an audio input device is connected to your computer for all programs.

# Setup
- Create and run a virtual environment (on Windows):
    ```
    py -m venv venv
    venv\Scripts\activate
    ```

- Install requirements:

    ```
    pip install -r requirements.txt
    ```

# Gathering Training Data
- Run the program with the following command:

    ```
    py gather_data.py
    ```

- In utils.py, set the NAME, NUMBER, and ACTIVITY variables
- Press any button on your DIPPID input device to start recording movement data. After 10 seconds, recording will stop on its own
- Acceleration and gyroscope data will be saved to a .csv file along with a timestamp and an ID for each row. The .csv file will be named after the variables in utils.py
- .csv files can be found in their respective subdirectories the data/ directory

# Activity Recognition
- Run the program with the following command:

    ```
    py fitness_trainer.py
    ```

- A pyglet window will open. Press any button on your DIPPID input device to start
- Once the workout is ready, use your DIPPID input device to follow along with the instructive images. If movements are performed correctly, the text on the screen will turn green
- Your total score will be displayed at the end of the workout