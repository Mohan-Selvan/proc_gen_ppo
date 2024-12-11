import pygame
import constants
import pickle
import math

from game_world import GameWorld
from constants import Direction
from constants import DIRECTIONS
from constants import GRID_SIZE

import numpy as np

from collections import deque
import os
import json

import ppo
import exporter

from sb3_contrib import RecurrentPPO
import custom_policy_lstm

if __name__ == "__main__":

    player_path = []
    with open (constants.DEFAULT_PLAYER_PATH_FILE_PATH, 'rb') as fp:
        player_path = pickle.load(fp)
        print("Path loaded from 'path_list'")

    env = GameWorld(width= GRID_SIZE[0], height= GRID_SIZE[1], 
                    player_path=player_path,
                    observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE, 
                    mask_shape=constants.ACTION_MASK_SHAPE, 
                    num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL, 
                    path_randomness=0.1, 
                    random_seed=2,
                    force_move_agent_forward=True)
    
    env.reset()

    #env.grid = np.full_like(env.grid, 1, dtype=int)

    # results_directory = "./saves/evaluation/proposed_approach/"
    # results = {}
    # with open(os.path.join(results_directory, "results.json"), "r") as file:
    #     results = json.load(file)

    # result = results[0]
    # env.grid = np.array(result["env_data"]["grid"])

    paths_data_file_path = "./saves/paths_data.json"
    paths_data = []
    with open(paths_data_file_path, "r") as file:
        paths_data = json.load(file)

    path_data = paths_data[76]
    path = path_data["path"]
    for index, cell in enumerate(path):
            path[index] = (cell[0], cell[1])
        
    env.set_player_path(path)

    while True:

        is_quit = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_quit = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_BACKSPACE:
                    is_quit = True
                
                else:

                    if(event.key == pygame.K_1):
                        model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, 
                        #policy_kwargs=dict(normalize_images=False, ortho_init=True, lstm_hidden_size=256),
                        policy_kwargs = dict(
                            normalize_images=False,
                            features_extractor_class=custom_policy_lstm.CustomSmallCnnFeatureExtractor,
                            features_extractor_kwargs=dict(features_dim=1024),
                        ),
                        gamma=0.99, 
                        #gae_lambda=0.95,
                        n_epochs=10, 
                        #ent_coef=0.1,
                        #clip_range=0.3,
                        #max_grad_norm=0.5,
                        #vf_coef=0.5,
                        learning_rate=3e-4,
                        normalize_advantage=True,
                        seed=constants.RANDOM_SEED)
                        # Print the full network architecture
                        print(model.policy)
                        print('______________________')
                        for name, param in model.policy.state_dict().items():
                            print(f"{param}")
                            print('--------------------------')

            if(event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]):
                position = pygame.mouse.get_pos()
                grid_pos = (position[0] // cell_draw_size, position[1] // cell_draw_size)

                print(f"Click grid pos : {grid_pos}")

                if(env.is_position_valid(grid_pos)):
                    env.grid[grid_pos] = constants.GRID_EMPTY_SPACE if env.grid[grid_pos] == constants.GRID_PLATFORM else constants.GRID_PLATFORM

            elif(event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed(3)[2]):
                position = pygame.mouse.get_pos()
                grid_pos = (position[0] // cell_draw_size, position[1] // cell_draw_size)

                print(f"Click grid pos : {grid_pos}")

                if(env.is_position_valid(grid_pos)):
                    env.grid[grid_pos] = constants.GRID_EMPTY_SPACE if env.grid[grid_pos] == constants.GRID_LAVA else constants.GRID_LAVA

        env._update(flip_display= False)

        pygame.image.save(env.display, "./player_path.png")
        break

        (draw_border) = False
        cell_draw_size = constants.CELL_DRAW_SIZE

        if(draw_border):
            for x in range(0, env.grid.shape[0]):
                for y in range(0, env.grid.shape[1]):
                        pygame.draw.rect(env.display, constants.COLOR_WHITE, rect= pygame.Rect((x * cell_draw_size), y * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size // 20, border_radius = 1)


        pygame.display.flip()

        if(is_quit):
            break
                
    print("Exiting game : 0")


