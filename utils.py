WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
NUMBER = 1
NAME = 'leeann'
ACTIVITIES = ['running', 'rowing', 'jumpingjack', 'lifting']
ACTION = ACTIVITIES[0]
FONT_NAME = 'Courier New'
FONT_COLOR = (0, 0, 0, 255)
ACTIVE_COLOR = (0, 255, 0, 255)
IMG_DIR = 'img/'
DIRECTORY = 'data/'
FILE_PATH = f'{DIRECTORY}{ACTION}/{NAME}-{ACTION}-{NUMBER}.csv'

LABEL_DICT = {
    "running": 0,
    "rowing": 1,
    "jumpingjack": 2,
    "lifting": 3
}