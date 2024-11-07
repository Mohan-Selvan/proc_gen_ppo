import pygame
import constants
import pickle

from game_world import GameWorld
from constants import Direction
from constants import DIRECTIONS
from constants import GRID_SIZE

import numpy as np

from collections import deque

if __name__ == "__main__":

    env = GameWorld(width= GRID_SIZE[0], height= GRID_SIZE[1], player_path=[(0, 0)],observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE, mask_shape=constants.ACTION_MASK_SHAPE, num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL)

    is_path_define_mode = True
    path_define_cell = env.start_pos
    env.set_player_path([path_define_cell])

    covered_path = []

    while True:

        is_quit = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_quit = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_BACKSPACE:
                    is_quit = True


                if(is_path_define_mode):

                    if(event.key == pygame.K_LEFT):
                        cell = env.get_cell_in_direction(path_define_cell, Direction.LEFT, restrict_boundary=True)
                        if(not (cell in env.player_path)):
                            env.player_path.append(cell)
                            path_define_cell = cell

                        if(cell == env.end_pos):
                            is_path_define_mode = False
                            

                    if(event.key == pygame.K_RIGHT):
                        cell = env.get_cell_in_direction(path_define_cell, Direction.RIGHT, restrict_boundary=True)
                        if(not (cell in env.player_path)):
                            env.player_path.append(cell)
                            path_define_cell = cell

                        if(cell == env.end_pos):
                            is_path_define_mode = False

                    if(event.key == pygame.K_UP):
                        cell = env.get_cell_in_direction(path_define_cell, Direction.UP, restrict_boundary=True)
                        if(not (cell in env.player_path)):
                            env.player_path.append(cell)
                            path_define_cell = cell

                        if(cell == env.end_pos):
                            is_path_define_mode = False

                    if(event.key == pygame.K_DOWN):
                        cell = env.get_cell_in_direction(path_define_cell, Direction.DOWN, restrict_boundary=True)
                        if(not (cell in env.player_path)):
                            env.player_path.append(cell)
                            path_define_cell = cell

                        if(cell == env.end_pos):
                            is_path_define_mode = False

                    if(not is_path_define_mode):
                        with open('./saves/path_list', 'wb') as fp:
                            pickle.dump(env.player_path, fp)

                        print("Path define mode exited, Saving path as 'path_list'")

                    if(event.key == pygame.K_l):
                        is_path_define_mode = False
                        with open ('./saves/path_list', 'rb') as fp:
                            path_list = pickle.load(fp)
                            env.set_player_path(path_list)
                            print("Path define mode exited, Existing path loaded from 'path_list'")

                    if(event.key == pygame.K_k):
                        path_list = env.generate_path_from_start_to_end()
                        env.set_player_path(path_list)
                        print("Updated path")

                else:

                    reward = None
                    terminated = None
                    truncated = None
                    new_state = None

                    if(event.key == pygame.K_1):

                        mask =np.array(
                            [[0, 1, 2, 1, 0],
                            [1, 2, 0, 1, 1],
                            [1, 2, 0, 0, 2],
                            [1, 0, 0, 1, 1],
                            [1, 2, 0, 1, 1],
                            ]).T

                        new_state, reward, terminated, truncated, _ = env.step(action=mask)
                        norm_frame_count = (env.frame_count / env.max_frame_count)
                        print(f"Frame : {env.frame_count}, NormalizedFrameCount : {norm_frame_count} Reward : {reward}, Terminated : {terminated}, Truncated : {truncated}")
                        #print(f"State : {new_state}")
                        print(env.player_pos)

                    if(event.key == pygame.K_2):
                        obs = env._get_obs()
                        print(f"Obs shape : {obs.shape}")
                        print(obs)
                        print(f"Act : {env.action_space}")

                    if(event.key == pygame.K_3):
                        percent, path = env.calculate_reachability(max_distance=6)
                        print(f"Reachability : {percent}")
                        env.coverable_path = path

                    if(event.key == pygame.K_4):
                        env.set_player_path(env.generate_player_path(randomness=0.1))
                        print("Generated player path")


                    if(event.key == pygame.K_k):

                        covered_path = env.find_path(env.start_pos, env.end_pos)
                        print(f"Cells : \n{covered_path}")
                        # percent, covered_path = env.player_path_coverage()
                        # print(f"Reach : {percent}")

        
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
        #     pygame.draw.rect(env.display, constants.COLOR_MAGENTA, rect= pygame.Rect(cell[0] * cell_draw_size, cell[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 0)

        # pygame.display.flip()

        if(is_quit):
            break
                
    print("Exiting game : 0")


