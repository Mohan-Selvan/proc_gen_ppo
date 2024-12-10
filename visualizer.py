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

import ppo
import exporter

from sb3_contrib import RecurrentPPO
import custom_policy_lstm

if __name__ == "__main__":

    env = GameWorld(width= GRID_SIZE[0], height= GRID_SIZE[1], 
                    player_path=[(0, 0), (1, 1)],
                    observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE, 
                    mask_shape=constants.ACTION_MASK_SHAPE, 
                    num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL, 
                    path_randomness=0.1, 
                    random_seed=2,
                    force_move_agent_forward=True)
    
    env.reset()

    covered_path = []

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
                            features_extractor_class=custom_policy_lstm.CustomCnnFeatureExtractor,
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

        env._update(flip_display= True)

        cell_draw_size = constants.CELL_DRAW_SIZE
        # for cell in covered_path:
        #     pygame.draw.rect(env.display, constants.COLOR_MAGENTA, rect= pygame.Rect(cell[0] * cell_draw_size, cell[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 8)

        # pygame.display.flip()

        if(is_quit):
            break
                
    print("Exiting game : 0")


