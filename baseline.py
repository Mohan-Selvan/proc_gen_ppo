import game_world
import constants
import pickle
import random
import json
import numpy as np
import pygame
import os

player_path = []
with open (constants.DEFAULT_PLAYER_PATH_FILE_PATH, 'rb') as fp:
    player_path = pickle.load(fp)
    print("Path loaded from 'path_list'")

def evaluate_baseline(export_directory):

    print("Starting model evaluation")

    # Create the environment
    env = game_world.GameWorld(width=constants.GRID_SIZE[0], 
                            height=constants.GRID_SIZE[1], 
                            player_path=player_path,
                            observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE,
                            mask_shape=constants.ACTION_MASK_SHAPE, 
                            num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL,
                            path_randomness=0.5,
                            random_seed=constants.RANDOM_SEED,
                            force_move_agent_forward=True
                            ) 

    results = []
    
    for index in range(0, len(paths_data)):
        
        id = (index + 1)
        path_data = paths_data[index]

        path = path_data["path"]

        for index, cell in enumerate(path):
             path[index] = (cell[0], cell[1])

        path_complexity = path_data["complexity"]
        print(f"Running path with complexity : {path_complexity}")

        env.set_player_path(path)
        obs, info = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            
            # Get model prediction - Stochastic
            action = np.random.randint( low = 0, 
                                        high = (constants.NUMBER_OF_ACTIONS_PER_CELL), # No -1, because in this function, parameter 'high' is exclusive. 
                                        size= (1, constants.ACTION_MASK_SHAPE[0] * constants.ACTION_MASK_SHAPE[1])
                                    )
            
            obs, reward, terminated, truncated, info = env.step(action)
            terminated = info["data"]["is_solvable"]

        image = info["img"]
        pygame.image.save(pygame.image.fromstring(image, constants.WINDOW_RESOLUTION, 'RGBA'), os.path.join(export_directory, f"test_path_{id}_img.png"))
        result = {"path_id" : id, "path_data" : path_data, "env_data" : info["data"]}

        is_solvable = info["data"]["is_solvable"]
        print(f"Is_Level_Solvable : {is_solvable}")
        
        results.append(result)

    
    results = {"levels" : results}

    results_file = os.path.join(export_directory, "results.json")
    with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Model evaluation complete, results stored as {results_file}")

    return results


if(__name__ == "__main__"):
    
    paths_data_file_path = "./saves/paths_data.json"
    paths_data = []
    with open(paths_data_file_path, "r") as file:
        paths_data = json.load(file)

    results_directory = "./saves/evaluation/approach_1/"
    os.makedirs(results_directory, exist_ok=True)

    results = evaluate_baseline(export_directory=results_directory)

    results = {}
    with open(os.path.join(results_directory, "results.json"), "r") as file:
        results = json.load(file)

    solvable_count = 0

    for r in results["levels"]:
         if(r["env_data"]["is_solvable"]):
              solvable_count += 1

    print(f"Solvable count : {solvable_count} / {len(results)}" )
