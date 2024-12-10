import numpy as np
import game_world
import json
import constants
import pygame
import os
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO

def evaluate_model(model, export_directory):

    print("Starting model evaluation")

    # Create the environment
    env = game_world.GameWorld(width=constants.GRID_SIZE[0], 
                            height=constants.GRID_SIZE[1], 
                            player_path=paths_data[0]["path"],
                            observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE,
                            mask_shape=constants.ACTION_MASK_SHAPE, 
                            num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL,
                            path_randomness=0.5,
                            random_seed=constants.RANDOM_SEED,
                            force_move_agent_forward=False
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
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)

        image = info["img"]
        pygame.image.save(pygame.image.fromstring(image, constants.WINDOW_RESOLUTION, 'RGBA'), os.path.join(export_directory, f"test_path_{id}_img.png"))
        result = {"path_id" : id, "path_data" : path_data, "env_data" : info["data"]}

        is_solvable = info["data"]["is_solvable"]
        print(f"Is_Level_Solvable : {is_solvable}")
        
        results.append(result)

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

    results_directory = "./saves/evaluation/proposed_approach"
    os.makedirs(results_directory, exist_ok=True)

    # model = RecurrentPPO.load("./saves/model.zip")
    # results = evaluate_model(model, export_directory=results_directory)

    results = {}
    with open(os.path.join(results_directory, "results.json"), "r") as file:
        results = json.load(file)

    solvable_count = 0

    path = paths_data[0]["path"]

    for index, cell in enumerate(path):
            path[index] = (cell[0], cell[1])


    env = game_world.GameWorld(width=constants.GRID_SIZE[0], 
            height=constants.GRID_SIZE[1], 
            player_path=path,
            observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE,
            mask_shape=constants.ACTION_MASK_SHAPE, 
            num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL,
            path_randomness=0.5,
            random_seed=constants.RANDOM_SEED,
            force_move_agent_forward=False
            )
    
    for r in results:
        if(r["env_data"]["is_solvable"]):
            solvable_count += 1

            env_data = r["env_data"]
            grid = env_data["grid"]

            env.grid = np.array(grid)
            env.render(flip_display=True)
            pygame.image.save(env.display, os.path.join("./saves/visualizations", f"test_img.png"))

    print(f"Solvable count : {solvable_count} / {len(results)}" )

