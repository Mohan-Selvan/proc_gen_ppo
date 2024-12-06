import game_world
import constants
import pickle
import random
import numpy as np

player_path = []
with open (constants.DEFAULT_PLAYER_PATH_FILE_PATH, 'rb') as fp:
    player_path = pickle.load(fp)
    print("Path loaded from 'path_list'")

def create_env():

    # Create the environment
    env = game_world.GameWorld(width=constants.GRID_SIZE[0], 
                            height=constants.GRID_SIZE[1], 
                            player_path=player_path,
                            observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE,
                            mask_shape=constants.ACTION_MASK_SHAPE, 
                            num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL,
                            path_randomness=0.5,
                            random_seed=constants.RANDOM_SEED
                            ) 

    return env

def run_baseline(num_episodes):

    env = create_env()
    env.set_player_path(player_path)

    for episode in num_episodes:
        action = np.random.randint(0, constants.NUMBER_OF_ACTIONS_PER_CELL - 1, 
                                   size= (1, constants.ACTION_MASK_SHAPE[0] * constants.ACTION_MASK_SHAPE[1]))
        env.step(action)