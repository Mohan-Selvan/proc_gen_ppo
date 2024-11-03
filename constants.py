from enum import Enum

GRID_SIZE = (36, 36)
GAME_SIMULATION_SPEED = 30

# WORLD 
COLOR_BLACK =   (0, 0, 0)
COLOR_WHITE =   (255, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_RED = (255, 0, 0)
COLOR_BLUE =    (0, 0, 255)
COLOR_CYAN =    (0, 255, 255)
COLOR_BROWN =   (210, 105, 30)
COLOR_GREEN =   (0, 255, 0)
COLOR_YELLOW = (255, 255, 0)

GRID_EMPTY_SPACE = 0
GRID_PLATFORM = 1

CELL_DRAW_SIZE = 20

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    TOP_RIGHT = 5
    DOWN_RIGHT = 6
    DOWN_LEFT = 7
    TOP_LEFT = 8

DIRECTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.TOP_RIGHT, Direction.DOWN_RIGHT, Direction.DOWN_LEFT, Direction.TOP_LEFT]


# MODEL
STATE_SPACE = (GRID_SIZE[0] * GRID_SIZE[1]) * 4
MASK_SIZE = (1, 1)
ACTION_SPACE = 9
NUMBER_OF_ACTIONS_PER_CELL = 3

TILE_ACTION_IGNORE = 2
TILE_ACTION_PLACE_EMPTY_SPACE = 0
TILE_ACTION_PLACE_PLATFORM = 1