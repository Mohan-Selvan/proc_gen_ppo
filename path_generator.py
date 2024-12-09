import game_world
import constants
import pickle
import helper
import math
import numpy as np
import random
import json

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
                            random_seed=constants.RANDOM_SEED,
                            force_move_agent_forward=False
                            ) 
    return env

def get_complexity_score(target_path):

    def calculate_turns(path):
        turns = 0
        for i in range(1, len(path) - 1):
            prev = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            next = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if prev != next:
                turns += 1
        return turns
    
    def calculate_horizontal_vertical_ratio(path):
        horizontal_moves = 0
        vertical_moves = 0
        for i in range(1, len(path)):
            if path[i][0] == path[i-1][0]:
                horizontal_moves += 1
            else:
                vertical_moves += 1
        return horizontal_moves, vertical_moves

    horizontal_count, vertical_count = calculate_horizontal_vertical_ratio(target_path)
    return len(target_path) + calculate_turns(target_path) + (horizontal_count) + (vertical_count * 1.5)


def generate_paths(count, seed):

    # Seeding randoms
    np.random.seed(seed)
    random.seed(seed)

    env = create_env()
    env.reset()

    paths_data = []

    i = 0
    print("Generating paths...")

    while (len(paths_data) < count):

        max_turns = ((i + 1) / 2)
        randomness=helper.lerp(0, 0.9, min(1, ((i) / count)))
        path = env.generate_player_path(max_turns, randomness)
        
        env.set_player_path(path)
        env.render(flip_display=True)

        paths_list = [data["path"] for data in paths_data]
        if(path not in paths_list):
            paths_data.append({
                "path" : path,
                "complexity" : get_complexity_score(path),
            })
            i += 1

    print("Generated paths")

    env.close()

    # Sorting paths based on complexity
    paths_data = sorted(paths_data, key=lambda x: x["complexity"])

    return paths_data
     
if(__name__ == "__main__"):
    paths_data = generate_paths(count=100, seed=constants.RANDOM_SEED)
    for data in paths_data:
        print("-----------------")
        print(data)
        print("-----------------")

    # Save to JSON
    output_file = "./saves/paths_data.json"
    with open(output_file, "w") as json_file:
        json.dump(paths_data, json_file, indent=4)
