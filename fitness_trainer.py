from activity_recognizer import Activity
import pyglet
from pyglet import window, clock
from random import shuffle
import os
import threading

from utils import WINDOW_WIDTH, WINDOW_HEIGHT, ACTIVITIES, IMG_DIR, FONT_NAME, FONT_COLOR, ACTIVE_COLOR

DURATION = 2
UPDATE_RATE = 0.1
PREP_TIME = 5.0

in_cooldown = False
cooldown = PREP_TIME
countdown = DURATION
score = 0
activities = ACTIVITIES.copy()
shuffle(activities)
current_activity = activities.pop()
user_activity = None

started = False
finished = False
activity = Activity()

win = window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
pyglet.gl.glClearColor(0.902, 0.961, 1.0, 1.0)

img1 = pyglet.sprite.Sprite(pyglet.image.load(f'{IMG_DIR}{current_activity}_1.png'), x=200, y=100)
img2 = pyglet.sprite.Sprite(pyglet.image.load(f'{IMG_DIR}{current_activity}_2.png'), x=400, y=100)
img1.scale = 0.2
img2.scale = 0.2

def update(dt):
    global activity, started, user_activity, current_activity, score
    if started:
        pred = activity.predict_live_data()
        if pred:
            user_activity = pred
            if user_activity == current_activity:
                score += dt
            print(f"Predicted activity: {pred}")


def count_down(dt):
    global countdown, cooldown, in_cooldown, current_activity, activities, finished, started
    if in_cooldown:
        if cooldown > 0:
            cooldown -= dt
        else:
            in_cooldown = False
    elif countdown > 0 and started and activity.finished_training and activity.got_live_data:
        countdown -= dt
    if countdown <= 0:
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


def train_model():
    global in_cooldown
    classifier, scaler = activity.train_classifier()
    activity.classifier = classifier
    activity.scaler = scaler
    in_cooldown = True


def update_images():
    global img1, img2, current_activity
    img1.image = pyglet.image.load(f'{IMG_DIR}{current_activity}_1.png')
    img2.image = pyglet.image.load(f'{IMG_DIR}{current_activity}_2.png')

def draw_start_screen():
    pyglet.text.Label('Welcome to FitnessTrainer', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label('Press any button to start your workout', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()

def draw_loading_screen():
    pyglet.text.Label('Preparing your workout...', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label('During workout, follow the instructions on the screen', font_name=FONT_NAME, font_size=15, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()


def draw_cooldown_screen():
    veil = pyglet.shapes.Rectangle(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, color=(100, 100, 100, 180))
    img1.draw()
    img2.draw()
    veil.draw()
    pyglet.text.Label(f'Next up: {current_activity}', font_name=FONT_NAME, font_size=30, color=(255, 255, 255), x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label(f'in {round(cooldown, 1)} seconds', font_name=FONT_NAME, font_size=18, color=(255, 255, 255), x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()

def draw_active_screen():
    activity_label = pyglet.text.Label(f'ACTIVITY: {current_activity}', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 80, anchor_x='center', anchor_y='center')
    pyglet.text.Label(f'{round(countdown, 1)} seconds', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH-200, y=WINDOW_HEIGHT-20, anchor_x='center').draw()
    activity_label.color = ACTIVE_COLOR if current_activity == user_activity else (255, 0 , 0, 255)
    activity_label.draw()
    img1.draw()
    img2.draw()

def draw_end_screen():
    pyglet.text.Label('You did it!', font_name=FONT_NAME, font_size=36, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 + 40, anchor_x='center', anchor_y='center').draw()
    pyglet.text.Label(f'You were on targt for {score/(4*DURATION):.2%} of your workout.', font_name=FONT_NAME, font_size=20, color=FONT_COLOR, x=WINDOW_WIDTH//2, y=WINDOW_HEIGHT//2 - 40, anchor_x='center', anchor_y='center').draw()


def handle_btn_press(data):
    global started, finished
    if int(data) == 1 and not started and not finished:
        started = True
        threading.Thread(target=train_model, daemon=True).start()

activity.sensor.register_callback('button_1', handle_btn_press)
activity.sensor.register_callback('button_2', handle_btn_press)
activity.sensor.register_callback('button_3', handle_btn_press)
    
@win.event
def on_draw():
    win.clear()
    if not started and not finished:
        draw_start_screen()
    elif started and not activity.finished_training and not activity.got_live_data:
        draw_loading_screen()
    elif in_cooldown:
        draw_cooldown_screen()
    elif started and activity.finished_training and activity.got_live_data:
        draw_active_screen()
    elif finished and not started:
        draw_end_screen()

@win.event
def on_close():
    os._exit(0)

pyglet.app.run()