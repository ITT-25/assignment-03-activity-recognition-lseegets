from activity_recognizer import Recognizer
import pyglet
from pyglet import window, clock
from random import shuffle
import os
import threading

from utils import WINDOW_WIDTH, WINDOW_HEIGHT, ACTIVITIES, IMG_DIR, FONT_NAME, FONT_COLOR, ACTIVE_COLOR

DURATION = 20      # How long one activity should be performed
UPDATE_RATE = 0.1   # Rate at which the pyglet window is updated
PREP_TIME = 5    # Time to prepare for the next activity (during cooldown)

started = False         # Checks if workout has started
finished = False        # Checks if workout has finished
in_cooldown = False     # Checks if cooldown is active
cooldown = PREP_TIME    
countdown = DURATION
score = 0               # Score to keep track of performance (will be displayed after workout)

activities = ACTIVITIES.copy()          # Make a copy of all available activities to shuffle them
shuffle(activities)
current_activity = activities.pop()     
user_activity = None                    # Predicted activity based on the sensor data

recognizer = Recognizer()   # The activity recognizer
win = window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
pyglet.gl.glClearColor(0.902, 0.961, 1.0, 1.0)  # background color

# The images for the activities
img1 = pyglet.sprite.Sprite(pyglet.image.load(f'{IMG_DIR}{current_activity}_1.png'), x=200, y=100)
img2 = pyglet.sprite.Sprite(pyglet.image.load(f'{IMG_DIR}{current_activity}_2.png'), x=400, y=100)
img1.scale = 0.2
img2.scale = 0.2


# Train the classifier upon starting the application

def on_start(dt):
    threading.Thread(target=recognizer.train_classifier, daemon=True).start()

clock.schedule_once(on_start, 0)


# Check if the incoming data from the DIPPID device matches the current activity

def update(dt):
    global recognizer, finished, user_activity, current_activity, score
    if recognizer.finished_training and not finished:
        pred = recognizer.predict_live_data()
        if pred:
            user_activity = pred
            if user_activity == current_activity:
                score += dt
            
            # For debugging: Print the predicted activity
            # print(f"Predicted activity: {pred}")


# Count down to time the activity. Add a cooldown phase between activities

def count_down(dt):
    global countdown, cooldown, in_cooldown, current_activity, activities, finished, started
    if in_cooldown:
        if cooldown > 0:
            cooldown -= dt
        else:
            in_cooldown = False
    elif countdown > 0 and started and recognizer.finished_training and recognizer.got_live_data:
        countdown -= dt

    if countdown <= 0:
        # Check if there are still activities to do. If so, update the current activity. If not, end the workout
        if activities:
            cooldown = PREP_TIME
            in_cooldown = True
            countdown = DURATION
            current_activity = activities.pop()
            update_images()
        else:
            finished = True
            started = False

clock.schedule_interval(update, UPDATE_RATE)
clock.schedule_interval(count_down, UPDATE_RATE)


# Update the sprites based on the current activity

def update_images():
    global img1, img2, current_activity
    img1.image = pyglet.image.load(f'{IMG_DIR}{current_activity}_1.png')
    img2.image = pyglet.image.load(f'{IMG_DIR}{current_activity}_2.png')


# Drawing the screens

def draw_start_screen():
    pyglet.text.Label('Welcome to FitnessTrainer!', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label('Press any button to start your workout.', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label('Follow the instructions once workout starts', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 120, anchor_x='center', anchor_y='center').draw()

def draw_loading_screen():
    pyglet.text.Label('Preparing workout data...', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label('This might take a few moments', font_name=FONT_NAME, font_size=15, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()

def draw_cooldown_screen():
    veil = pyglet.shapes.Rectangle(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, color=(100, 100, 100, 180))
    img1.draw()
    img2.draw()
    veil.draw()
    pyglet.text.Label(f'Next up: {current_activity}', font_name=FONT_NAME, font_size=30, color=(255, 255, 255), x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label(f'in {round(cooldown, 1)} seconds', font_name=FONT_NAME, font_size=18, color=(255, 255, 255), x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()

def draw_active_screen():
    activity_label = pyglet.text.Label(f'ACTIVITY: {current_activity}', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 80, anchor_x='center', anchor_y='center')
    pyglet.text.Label(f'{round(countdown, 1)} seconds', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH-100, y=WINDOW_HEIGHT-20, anchor_x='center').draw()
    activity_label.color = ACTIVE_COLOR if current_activity == user_activity else (255, 0 , 0, 255)
    activity_label.draw()
    img1.draw()
    img2.draw()

def draw_end_screen():
    pyglet.text.Label('You did it!', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label(f'You were on target for {score/(len(ACTIVITIES)*DURATION):.2%} of your workout.', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()


# Registering button presses to start the workout

def handle_btn_press(data):
    global started, finished, in_cooldown
    if int(data) == 1 and not started and not finished:
        started = True
        in_cooldown = True

recognizer.sensor.register_callback('button_1', handle_btn_press)
recognizer.sensor.register_callback('button_2', handle_btn_press)
recognizer.sensor.register_callback('button_3', handle_btn_press)
    

@win.event
def on_draw():
    win.clear()
    if not started and (not recognizer.finished_training or not recognizer.got_live_data):
        draw_loading_screen()
    elif not started and not finished and recognizer.finished_training and recognizer.got_live_data:
        draw_start_screen()
    elif in_cooldown:
        draw_cooldown_screen()
    elif started and recognizer.finished_training and recognizer.got_live_data:
        draw_active_screen()
    elif finished and not started:
        draw_end_screen()


# Make sure the program will actually stop upon closing the window

@win.event
def on_close():
    os._exit(0)

pyglet.app.run()