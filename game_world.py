import pygame
import numpy as np
import math
import random
import helper
import pickle

import constants

import heapq

from constants import Direction
from collections import deque
from typing import Optional

import gymnasium as gym
import sys
import exporter

np.set_printoptions(threshold=sys.maxsize)

class GameWorld(gym.Env):

    def __init__(self, width, height, player_path, observation_window_shape, mask_shape, num_tile_actions, path_randomness, random_seed, force_move_agent_forward = False):

        self.width = width
        self.height = height

        self.screen_resolution_X, self.screen_resolution_Y = 1024, 768
        self.screen_resolution = (self.screen_resolution_X, self.screen_resolution_Y)

        self.display = pygame.display.set_mode(self.screen_resolution)
        pygame.display.set_caption("Game")
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.grid = np.zeros([self.width, self.height], np.uint8)

        self.start_pos = (7, self.height // 2)
        self.end_pos = (self.width - 8, self.height // 2)

        self.player_path_index = 0
        self.player_pos = self.start_pos
        self.coverable_path = []

        path_list = [self.start_pos]
        self.mask = np.full(mask_shape, 0, dtype=np.uint8)

        self.player_path = path_list
        self.max_frame_count = 10000
        self.iterations_per_game = 1
        self.path_randomness = path_randomness
        self.max_distance_from_path = 6
        self.reset_count = 0
        self.total_number_of_possible_cells_that_can_be_modified = 1
        self.step_count = 0
        self.max_step_count = 1000
        self.force_move_agent_forward = force_move_agent_forward

        self.observation_window_shape = observation_window_shape
        self.mask_shape = mask_shape
        self.num_tile_actions = num_tile_actions

        # History
        self.last_action_mask = np.full(self.mask_shape, fill_value=constants.GRID_PLATFORM, dtype=np.uint8)
        self.last_player_pos = self.player_pos

        # Trackers
        self.train_level_count = 0

        # Action space: Each element in the 2D mask has 3 possible values (0, 1, or 2)
        # self.action_space = gym.spaces.MultiDiscrete(([num_tile_actions] * (mask_shape[0] * mask_shape[1])), seed=random_seed, dtype=np.uint8
        # )

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, mask_shape[0] * mask_shape[1]), seed=random_seed, dtype=np.float32
        )

        # Observation space: (3 channels, grid_size X, grid_size Y)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, self.observation_window_shape[0], self.observation_window_shape[1]), seed=random_seed, dtype=np.float32
        )

        self.set_player_path(player_path)
        # self.reset()
    
    def _get_obs(self):
      
        # normalized_grid = self.grid.copy()
        # normalized_grid = (normalized_grid / (constants.TOTAL_NUMBER_OF_TILE_TYPES - 1))

        # _, reachable_cells, _ = self.calculate_reachability(max_distance=self.max_distance_from_path) 
        # ohe_grid_reachable_cells = np.zeros_like(self.grid, dtype=np.uint8)        
        # for cell in reachable_cells:
        #     ohe_grid_reachable_cells[cell] = 1

        # ohe_grid_player_path = np.zeros_like(self.grid, dtype=np.uint8)        
        # for cell in self.player_path:
        #     ohe_grid_player_path[cell] = 1
        
        # # ohe_grid_player_pos = np.zeros_like(self.grid, dtype=np.uint8)
        # # ohe_grid_player_pos[self.player_pos] = 1
        # ohe_grid_mask_pos = np.zeros_like(self.grid, dtype=np.uint8)
        # ohe_grid_mask_pos[self.player_pos] = 1

        # window_shape = constants.ACTION_MASK_SHAPE
        # window = np.full(window_shape, 0, dtype = np.uint8)
        # window_grid = np.full_like(self.grid, 0, dtype = np.uint8)

        # base_pos = (self.player_pos[0] - (math.floor(window_shape[0] / 2)), self.player_pos[1] - math.floor(window_shape[1] / 2))
        # for x in range(0, window_shape[0]):
        #     for y in range(0, window_shape[1]):
        #         world_pos = base_pos[0] + x, base_pos[1] + y
        #         if(self.is_position_valid(world_pos)):
        #             window[x, y] = self.grid[world_pos]
        #             window_grid[world_pos] = self.grid[world_pos]
        #             ohe_grid_mask_pos[world_pos] = 1
   
        # state = np.stack([normalized_grid, ohe_grid_player_path, ohe_grid_mask_pos, ohe_grid_reachable_cells], axis=0).astype(np.float64) #  window_grid * 255
        # #obs = state.transpose(1, 2, 0) # Shape to (grid_size X, grid_size Y, 4 channels)
        # obs = state
        # return obs

        window_shape = self.observation_window_shape

        window_normalized_grid = np.full(window_shape, constants.GRID_PLATFORM, dtype = np.uint8)
        window_ohe_player_path = np.full(window_shape, 0, dtype = np.float32)
        window_ohe_reachable_points = np.full(window_shape, 0, dtype=np.uint8)
        window_ohe_mask_pos = np.full(window_shape, 0, dtype = np.uint8)
        window_last_mask_values = np.full(window_shape, constants.TILE_ACTION_IGNORE, dtype=np.uint8)

        # Current Mask positions
        mask_positions = []
        base_pos = (self.player_pos[0] - (math.floor(self.mask_shape[0] / 2)), self.player_pos[1] - math.floor(self.mask_shape[1] / 2))
        for x in range(0, self.mask_shape[0]):
            for y in range(0, self.mask_shape[1]):
                world_pos = base_pos[0] + x, base_pos[1] + y
                if(self.is_position_valid(world_pos)):
                    mask_positions.append(world_pos)

        # Last mask positions
        last_mask_dict = {}
        base_pos = (self.last_player_pos[0] - (math.floor(self.mask_shape[0] / 2)), self.last_player_pos[1] - math.floor(self.mask_shape[1] / 2))
        for x in range(0, self.mask_shape[0]):
            for y in range(0, self.mask_shape[1]):
                world_pos = base_pos[0] + x, base_pos[1] + y
                if(self.is_position_valid(world_pos)):
                    last_mask_dict[world_pos] = self.last_action_mask[x, y]

        base_pos = (self.player_pos[0] - (math.floor(window_shape[0] / 2)), self.player_pos[1] - math.floor(window_shape[1] / 2))
        for x in range(0, window_shape[0]):
            for y in range(0, window_shape[1]):
                world_pos = base_pos[0] + x, base_pos[1] + y
                
                if(self.is_position_valid(world_pos)):
                    window_normalized_grid[x, y] = self.grid[world_pos]
                if(world_pos in self.player_path):
                    window_ohe_player_path[x, y] = 1 if ((self.player_path.index(world_pos)) >= self.player_path_index) else 0.5
                if(world_pos in self.coverable_path):
                    window_ohe_reachable_points[x, y] = 1
                if(world_pos in mask_positions):
                    window_ohe_mask_pos[x, y] = 1
                if(world_pos in last_mask_dict):
                    window_last_mask_values[x, y] = last_mask_dict[world_pos]
                

        window_normalized_grid = (window_normalized_grid / (constants.TOTAL_NUMBER_OF_TILE_TYPES - 1))
        window_last_mask_values = (window_last_mask_values / (constants.NUMBER_OF_ACTIONS_PER_CELL - 1))
   
        state = np.stack([window_normalized_grid.astype(np.float32), 
                          window_ohe_player_path.astype(np.float32), 
                          window_ohe_reachable_points.astype(np.float32), 
                          #window_ohe_mask_pos.astype(np.float64), 
                          #window_last_mask_values.astype(np.float64)
                        ],axis=0) #window_ohe_reachable_points * 255
        
        # print(state)

        obs = state
        return obs
    
    def _get_info(self):
        return {
            "path_progress": (self.player_path_index / len(self.player_path))
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)

        # Seeding randoms
        np.random.seed(seed)
        random.seed(seed)

        self.frame_count = 0
        self.step_count = 0
        self.grid = np.full([self.width, self.height], constants.GRID_PLATFORM, np.uint8)
        self.player_pos = self.start_pos
        self.player_path_index = 0

        self.reset_count += 1
        # print(f"Game reset : {self.reset_count}")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        # print(action.shape)
        # print(action)

        # Convert the flat action back to a 2D action mask
        remapped_action = helper.remap_array(action, (-1.0, 1.0), (0, self.num_tile_actions - 1))
        # print(remapped_action)
        remapped_action = np.round(remapped_action).astype(int)
        # print(remapped_action)
        action_mask = remapped_action.reshape((self.mask_shape[0], self.mask_shape[1]))
        # print(action_mask)

        # print("----------------------------------")
        #action_mask = np.round(np.array(action).reshape((self.mask_shape[0], self.mask_shape[1]))).astype(int)
        # print(action_mask.shape)
        # print(action_mask)
        # print("--------------------------")

        # print(action_mask_2d.shape)
        # print(action_mask_2d)

        pygame.event.get()
        
        self._update(False)
        # reward_before_action = self.get_current_reward()

        # Deciding action
        self.mask = action_mask
        self.execute_tile_mask(self.mask)
        self.last_action_mask = action_mask
        
        state = self._get_obs()
        # reward_after_action = self.get_current_reward()
        # reward = (reward_after_action - reward_before_action)

        # print(f"Reward, B: {reward_before_action} A: {reward_after_action}")

        #terminated = self.frame_count > self.max_frame_count
        terminated = self.step_count > self.max_step_count
        truncated = False

        # Checking if any reachable points are in mask area
        # found = False
        # pivotX, pivotY = (math.floor(self.mask_shape[0] / 2), math.floor(self.mask_shape[1] / 2))
        # posX, posY = self.player_pos
        # for x in range(0, self.mask_shape[0]):
        #     for y in range(0, self.mask_shape[1]):
        #         grid_pos = posX + x - pivotX, posY + y - pivotY
        #         if(grid_pos in reachable_cells):
        #             found = True
        #             break
        # truncated = not found

        # if(not found):
        #     penalty = ((1.0 - reachability_percent))
        #     reward -= penalty
        #     print(f"Reachability : {reachability_percent}, Penalty : {penalty}")

        ##############################################################################

        reachability_percent, reachable_cells, highest_reachable_path_index = self.calculate_reachability(max_distance=self.max_distance_from_path)
        # if(highest_reachable_path_index < self.player_path_index - 10):
        #     # reward = 0
        #     truncated = True

        self.coverable_path = reachable_cells

        # Cells in action mask
        cells_in_action_mask = self.get_cells_in_action_mask_region()

        # Finding the furtherest player path cell in the action mask region.
        furthest_cell_action_mask = None
        for cell in reversed(self.player_path):
            if(cell in cells_in_action_mask):
                furthest_cell_action_mask = cell
                break
        
        is_furthest_cell_in_action_mask_reachable = (not (furthest_cell_action_mask is None)) and (furthest_cell_action_mask in reachable_cells)

        if(is_furthest_cell_in_action_mask_reachable):
            
            reward = 5

            # reward for non_hanging cells
            hanging_cells = []
            for cell in cells_in_action_mask:
                if(self.is_cell_hanging(cell)):
                    hanging_cells.append(cell)
            reward += (1.0 - ((len(hanging_cells) / len(cells_in_action_mask)))) * 5

            self.step_count = 0

        else:
            reward = 0

        # Reduce reward if the step_count is greater than half of max.
        reward -= max(0, ((self.step_count - (self.max_step_count*0.5)) / self.max_step_count))

        
        found = False
        pivotX, pivotY = (math.floor(self.mask_shape[0] / 2), math.floor(self.mask_shape[1] / 2))
        posX, posY = self.player_pos
        for x in range(0, self.mask_shape[0]):
            for y in range(0, self.mask_shape[1]):
                grid_pos = posX + x - pivotX, posY + y - pivotY
                if(grid_pos in reachable_cells):
                    found = True
                    break
        
        if(not found):
            reward -= 0

        if(is_furthest_cell_in_action_mask_reachable):
            self.last_player_pos = self.player_pos
                
            if(self.player_path_index >= len(self.player_path) - 1):
                terminated = True

            self.player_path_index = (self.player_path_index + 5)
            if(self.player_path_index >= len(self.player_path)):
                self.player_path_index = (len(self.player_path) - 1)
            self.player_pos = self.player_path[self.player_path_index]

        ##############################################################################

        # reward = 0

        # cells_in_action_mask = self.get_cells_in_action_mask_region()
        # for cell in cells_in_action_mask:
        #     if(self.grid[cell] == constants.GRID_EMPTY_SPACE and (cell in self.player_path)):
        #         reward += 0.2
        #     if(self.grid[cell] == constants.GRID_PLATFORM and (cell not in self.player_path)):
        #         reward += 0.1
        #     else:
        #         reward -= 0

        # player_path_cells_in_action_mask = []
        # for cell in cells_in_action_mask:
        #     if(cell in self.player_path):
        #         player_path_cells_in_action_mask.append(cell)

        # is_all_pp_cell_empty = True
        # for cell in player_path_cells_in_action_mask:
        #     if(self.grid[cell] != constants.GRID_EMPTY_SPACE):
        #         is_all_pp_cell_empty = False

        # # Move agent along path
        # if(is_all_pp_cell_empty):
        #     self.last_player_pos = self.player_pos
        #     self.player_path_index = (self.player_path_index + 5) % len(self.player_path)
        #     self.player_pos = self.player_path[self.player_path_index]

        ##############################################################################


        self.render(True)
        self.frame_count += 1
        self.step_count += 1

        if(terminated or truncated):
            reachability, _, highest_reachable_path_index = self.calculate_reachability(max_distance=self.max_distance_from_path)
            if(reachability >= 0.99):

                # Exporting valid levels.
                self.train_level_count += 1
                self.save_screen_image(f"./saves/train_levels/level_{self.train_level_count}_img.png")
                with open(f'./saves/train_levels/level_{self.train_level_count}_path', 'wb') as fp:
                    np.save(fp, self.player_path)
                with open(f'./saves/train_levels/level_{self.train_level_count}_grid', 'wb') as fp:
                    np.save(fp, self.grid)
                with open(f'./saves/train_levels/level_{self.train_level_count}_data', 'wb') as fp:
                    exporter.export_env(self, f'./saves/train_levels/level_{self.train_level_count}_data.json')
                
                # Randomizing player path.
                self.set_player_path(self.generate_player_path(max_turns=1000, randomness=self.path_randomness))
                print("Randomizing path")        

        return state, reward, terminated, truncated, self._get_info() 

    def get_current_reward(self):

        if(len(self.player_path) == 0):
            print("PLAYER PATH IS EMPTY!!")
            reward = -10000
            return reward

        reward, reachable_cells, highest_reachable_path_index = self.calculate_reachability(max_distance=self.max_distance_from_path)
        self.coverable_path = reachable_cells

        reward *= 5

        hanging_cells = self.find_hanging_cells()
        reward += ((1.0 - (len(hanging_cells) / self.total_number_of_possible_cells_that_can_be_modified)) * 2)


        def can_stand_on(cell):
            below_cell = self.get_cell_in_direction(cell, direction=Direction.DOWN, restrict_boundary=False)
            if(not self.is_position_valid(below_cell)):
                return False
            return (self.grid[cell] == constants.GRID_EMPTY_SPACE) and (self.grid[below_cell] == constants.GRID_PLATFORM)
        

        invalid_cells = []
        for x in range(0, self.width):
            for y in range(0, self.height):
                cell = (x, y)
                if((cell not in reachable_cells) and can_stand_on(cell)):
                    invalid_cells.append(cell)

        reward += ((1.0 - (len(invalid_cells) / self.total_number_of_possible_cells_that_can_be_modified)) * 3)

        reward += (self.player_path_index / len(self.player_path)) * 4

        return reward

        # # Checking if lava tiles are surrounded with proper cells
        # for x in range(0, self.width):
        #     for y in range(0, self.height):
        #         cell = (x, y)

        #         is_reduce_reward = False

        #         if(self.grid[cell] == constants.GRID_LAVA):
        #             for d in [Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
        #                 neighbor_cell = self.get_cell_in_direction(cell, direction=d,restrict_boundary=False)
        #                 if(not self.is_position_valid(neighbor_cell)):
        #                     continue
        #                 if(self.grid[neighbor_cell] == constants.GRID_EMPTY_SPACE):
        #                     is_reduce_reward = True
        #                     break
                    
        #             neighbor_cell = self.get_cell_in_direction(cell, direction=Direction.UP, restrict_boundary=False)
        #             if((self.is_position_valid(neighbor_cell)) and self.grid[neighbor_cell]  == constants.GRID_PLATFORM):
        #                 is_reduce_reward = True

        #         if(is_reduce_reward):
        #             reward -= 0.1

        # lava_tile_count = (self.grid == constants.GRID_LAVA).sum()
        # reward = reward - (min(0, lava_tile_count - 10))

        # for x in range(0, self.width):
        #     for y in range(0, self.height):
        #         if((x, y) in self.player_path):
        #             continue
        #         if(self.grid[x, y] == constants.GRID_PLATFORM):
        #             reward += 1
            
        # reward /= ((self.width * self.height) - len(self.player_path))
        # reward *= 2

        # blocks = 0
        # for cell in self.player_path:
        #     if(self.grid[cell] == constants.GRID_PLATFORM):
        #         blocks += 1

        # reward += (20 * (1.0 - (blocks / len(self.player_path))))

        # if(self.grid[self.start_pos] != constants.GRID_EMPTY_SPACE):
        #     reward -= 5000

        # if(self.grid[self.end_pos] != constants.GRID_EMPTY_SPACE):
        #     reward -= 5000

        return reward

    def execute_tile_mask(self, mask_2d):

        pivotX, pivotY = (math.floor(mask_2d.shape[0] / 2), math.floor(mask_2d.shape[1] / 2))
        posX, posY = self.player_pos
        for x in range(0, mask_2d.shape[0]):
            for y in range(0, mask_2d.shape[1]):
                grid_pos = posX + x - pivotX, posY + y - pivotY
                if(self.is_position_valid(grid_pos)):
                    if(mask_2d[x, y] == constants.TILE_ACTION_IGNORE):
                       continue
                    elif(mask_2d[x, y] == constants.TILE_ACTION_PLACE_EMPTY_SPACE): 
                        self.grid[grid_pos] = constants.GRID_EMPTY_SPACE
                    elif(mask_2d[x, y] == constants.TILE_ACTION_PLACE_PLATFORM): 
                        self.grid[grid_pos] = constants.GRID_PLATFORM
                    elif(mask_2d[x, y] == constants.TILE_ACTION_PLACE_LAVA): 
                        self.grid[grid_pos] = constants.GRID_LAVA
                    else:
                        print("INVALID TILE ACTION!")
      
    def is_position_valid(self, position):
        x, y = position
        return x >= 0 and x < self.width and y >= 0 and y < self.height
    
    def get_cell_in_direction(self, cell, direction, restrict_boundary = False):
        x, y = cell
        if(direction == Direction.UP):
            y -= 1

        elif(direction == Direction.RIGHT):
            x += 1

        elif(direction == Direction.DOWN):
            y += 1

        elif(direction == Direction.LEFT):
            x -= 1        

        elif(direction == Direction.TOP_RIGHT):
            x += 1
            y -= 1
        elif(direction == Direction.DOWN_RIGHT):
            x += 1
            y += 1
        elif(direction == Direction.DOWN_LEFT):
            x -= 1
            y += 1
        elif(direction == Direction.TOP_LEFT):
            x -= 1
            y -= 1

        if(restrict_boundary and (not self.is_position_valid((x, y)))):
            return cell
        
        return (x, y)

    def set_player_path(self, path_list):

        self.player_path = path_list
        self.player_pos = self.start_pos
        self.max_frame_count = (len(self.player_path) - 2) * self.iterations_per_game

        unique_cells = []

        for path_cell in self.player_path:

            posX, posY = path_cell
            pivotX, pivotY = (math.floor(self.mask_shape[0] / 2), math.floor(self.mask_shape[1] / 2))
            for local_X in range(0, self.mask_shape[0]):
                for local_Y in range(0, self.mask_shape[1]):
                    x, y = posX + local_X - pivotX, posY + local_Y - pivotY
                    cell = (x, y)
                    if(cell not in unique_cells):
                        unique_cells.append(cell)

        self.total_number_of_possible_cells_that_can_be_modified = len(unique_cells)
        print(f"Total number of cells that can be modified this level : {self.total_number_of_possible_cells_that_can_be_modified}")




    def _update(self, flip_display = True):
        self.clock.tick(constants.GAME_SIMULATION_SPEED)
        self.render(flip_display)

    def render(self, flip_display):

        self.display.fill(constants.COLOR_BLACK)
        cell_draw_size = constants.CELL_DRAW_SIZE

        for x in range(0, self.grid.shape[0]):
            for y in range(0, self.grid.shape[1]):
                if  (self.grid[x][y] == constants.GRID_EMPTY_SPACE):
                    pygame.draw.rect(self.display, constants.COLOR_BLACK, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= 1, border_radius = 1)
                elif(self.grid[x][y] == constants.GRID_PLATFORM):
                    pygame.draw.rect(self.display, constants.COLOR_BROWN, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 1)
                elif(self.grid[x][y] == constants.GRID_LAVA):
                    pygame.draw.rect(self.display, constants.COLOR_RED, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 1)
                else:
                    pygame.draw.rect(self.display, constants.COLOR_MAGENTA, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 1)

        pygame.draw.rect(self.display, constants.COLOR_PURPLE, rect= pygame.Rect(self.player_pos[0] * cell_draw_size, self.player_pos[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 1)
        pygame.draw.rect(self.display, constants.COLOR_GREEN, rect= pygame.Rect(self.start_pos[0] * cell_draw_size, self.start_pos[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 1)
        pygame.draw.rect(self.display, constants.COLOR_CYAN, rect= pygame.Rect(self.end_pos[0] * cell_draw_size, self.end_pos[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 1)

        # Render action mask placement
        posX, posY = self.player_pos
        cell_draw_size = constants.CELL_DRAW_SIZE
        mask_to_draw = self.mask
        pivotX, pivotY = (math.floor(mask_to_draw.shape[0] / 2), math.floor(mask_to_draw.shape[1] / 2))
        for local_X in range(0, mask_to_draw.shape[0]):
            for local_Y in range(0, mask_to_draw.shape[1]):
                x, y = posX + local_X - pivotX, posY + local_Y - pivotY
                pygame.draw.rect(self.display, constants.COLOR_BLUE, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= 1, border_radius = 1)

                color = constants.COLOR_BLACK
                mask_value = mask_to_draw[local_X, local_Y]
                
                if(mask_value == constants.TILE_ACTION_IGNORE):
                    color = constants.COLOR_BLACK
                elif(mask_value == constants.TILE_ACTION_PLACE_EMPTY_SPACE):
                    color = constants.COLOR_RED
                    pygame.draw.rect(self.display, color, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 8)
                elif(mask_value == constants.TILE_ACTION_PLACE_PLATFORM):
                    color = constants.COLOR_BROWN
                    pygame.draw.rect(self.display, color, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 8)
                elif(mask_value == constants.TILE_ACTION_PLACE_LAVA):
                    color = constants.COLOR_RED
                    pygame.draw.rect(self.display, color, rect= pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 8)

                

        for index, cell in enumerate(self.player_path):
            color = helper.lerp_color(constants.COLOR_GREEN, constants.COLOR_CYAN, (index + 1) / (len(self.player_path)))
            pygame.draw.rect(self.display, color, rect= pygame.Rect(cell[0] * cell_draw_size, cell[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 0)

        for cell in self.coverable_path:
            pygame.draw.rect(self.display, constants.COLOR_MAGENTA, rect= pygame.Rect(cell[0] * cell_draw_size, cell[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 2, border_radius = 8)

        if(flip_display):
            pygame.display.flip()

    def close(self):
        #print("Closing game")    
        pass

    def calculate_reachability(self, max_distance):

        def manhattan_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Check proximity for each player path cell
        def within_distance(cell, max_dist):
            return any(manhattan_distance(cell[0], cell[1], px, py) <= max_dist for px, py in player_path)
        
        def is_any_route_clear(cell, routes):
            is_clear = False
            for route in routes:
                if(are_route_cells_clear(cell, route)):
                    is_clear = True
                    break
            return is_clear
        
        def are_route_cells_clear(cell, directions):
            for (dx, dy) in directions:
                nx, ny = (cell[0] + dx, cell[1] + dy)
                new_cell = (nx, ny)

                if(not self.is_position_valid(new_cell)):
                    return False

                if(not (self.grid[new_cell] == constants.GRID_EMPTY_SPACE)):
                    return False

            return True

        
        def can_stand_on(cell):
            below_cell = self.get_cell_in_direction(cell, direction=Direction.DOWN, restrict_boundary=False)
            if(not self.is_position_valid(below_cell)):
                return False
            return (self.grid[cell] == constants.GRID_EMPTY_SPACE) and (self.grid[below_cell] == constants.GRID_PLATFORM)
        
        player_path = self.player_path

        reachable_cells = set()
        start_cell = player_path[0]

        # directions_and_routes = [
        #     ((-3, -3), []), ((-2, -3), []), ((-1, -3), []), ((0, -3), []), ((1, -3), []), ((2, -3), []), ((3, -3), []),
        #     ((-3, -2), []), ((-2, -2), []), ((-1, -2), []), ((0, -2), []), ((1, -2), []), ((2, -2), []), ((3, -2), []),
        #     ((-3, -1), []), ((-2, -1), []), ((-1, -1), []), ((0, -1), []), ((1, -1), []), ((2, -1), []), ((3, -1), []),
        #     ((-3,  0), []), ((-2,  0), []), ((-1,  0), []), ((0,  0), []), ((1,  0), []), ((2,  0), []), ((3,  0), []),
        #     ((-3,  1), []), ((-2,  1), []), ((-1,  1), []), ((0,  1), []), ((1,  1), []), ((2,  1), []), ((3,  1), []),
        #     ((-3,  2), []), ((-2,  2), []), ((-1,  2), []), ((0,  2), []), ((1,  2), []), ((2,  2), []), ((3,  2), []),
        #     ((-3,  3), []), ((-2,  3), []), ((-1,  3), []), ((0,  3), []), ((1,  3), []), ((2,  3), []), ((3,  3), []),
        #     ]  # Left, Right, Down for basic movement

        directions_and_routes = [
            # ((-3, -3), [[]]), 
            
            ((-2, -3), [
                [(0, -1), (0, -2), (0, -3), (-1, -3)],
                [(-1, 0), (0, -1), (-1, -1), (-1, -2), (-1, -3)]
            ]), 
            
            ((-1, -3), [
                [(0, -1), (0, -2), (0, -3)]
            ]),
            
            # ((0, -3), [[]]), 
            
            ((1, -3), [
                [(0, -1), (0, -2), (0, -3)]
            ]),
            
            ((2, -3), [
                [(0, -1), (0, -2), (0, -3), (1, -3)],
                [(1, 0), (0, -1), (1, -1), (1, -2), (1, -3)]
            ]), 
            
            # ((3, -3), [[]]),
            
            # ((-3, -2), [[]]),
            
            ((-2, -2), [
                [(0, -1), (0, -2), (0, -3), (-1, -3), (-2, -3)],
                [(0, -1), (-1, -1), (-1, -2)]
            ]), 
            
            ((-1, -2), [
                [(0, -1), (0, -2)]
            ]), 
            
            # ((0, -2), [[]]), 
            
            
            ((1, -2), [
                [(0, -1), (0, -2)]
            ]), 
            
            ((2, -2), [
                [(0, -1), (0, -2), (0, -3), (1, -3), (2, -3)],
                [(0, -1), (1, -1), (1, -2)]
            ]),
            
            # ((3, -2), [[]]),
            # ((-3, -1), [[]]), 
            
            ((-2, -1), [
                [(0, -1), (0, -2), (-1, -2), (-2, -2)]
            ]), 
            
            ((-1, -1), [
                [(0, -1)]
            ]), 
            
            # ((0, -1), [[]]), 
            
            ((1, -1), [
                [(0, -1)]
            ]), 
            
            ((2, -1), [
                [(0, -1), (0, -2), (1, -2), (2, -2)]
            ]), 
            
            # ((3, -1), [[]]),
            

            ((-3,  0), [
                [(0, -1), (-1, -1), (-1, -2), (-2, -2), (-2, -1), (-3, -1)]
            ]), 
            
            ((-2,  0), [
                [(0, -1), (-1, -1), (-2, -1)]
            ]), 
            
            ((-1,  0), [[(-1, 0)]]), 
            
            ((0,  0), [[(0, 0)]]), 
            
            ((1,  0), [[(1, 0)]]), 
            
            ((2,  0), [
                [(0, -1), (1, -1), (2, -1)]
            ]), 
            
            ((3,  0), [
                [(0, -1), (1, -1), (1, -2), (2, -2), (2, -1), (3, -1)]
            ]),

            # ((-3,  1), [[]]), 
            
            ((-2,  1), [
                [(0, -1), (-1, -1), (-2, -1), (-2, 0)]
            ]), 
            
            ((-1,  1), [
                [(-1, 0)]
            ]), 
            
            # ((0,  1), [[]]), 
            
            ((1,  1), [
                [(1, 0)]
            ]),
            
            ((2,  1), [
                [(0, -1), (1, -1), (2, -1), (2, 0)]
            ]), 
            
            # ((3,  1), [[]]),
            # ((-3,  2), [[]]), 
            
            
            ((-2,  2), [
                [(0, -1), (-1, -1), (-2, -1), (-2, 0), (-2, 1)]
            ]),
            
            ((-1,  2), [
                [(-1, 0), (-1, 1)]
            ]),
             
            # ((0,  2), [[]]), 
            
            ((1,  2), [
                [(1, 0), (1, 1)]
            ]), 
            
            ((2,  2), [
                [(0, -1), (1, -1), (2, -1), (2, 0), (2, 1)]
            ]),
            
            # ((3,  2), [[]]),
            # ((-3,  3), [[]]), 
            
            ((-2,  3), [
                [(0, -1), (-1, -1), (-2, -1), (-2, 0), (-2, 1), (-2, 2)],
            ]), 
            
            ((-1,  3), [
                [(-1, 0), (-1, 1), (-1, 2)]
            ]), 
            
            # ((0,  3), [[]]), 
            
            ((1,  3), [
                [(1, 0), (1, 1), (1, 2)]
            ]),
            
            ((2,  3), [
                [(0, -1), (1, -1), (2, -1), (2, 0), (2, 1), (2, 2)],
            ]),
            
            # ((3,  3), [[]]),
            ]

        if(self.grid[start_cell] != constants.GRID_EMPTY_SPACE):
            return 0, list(reachable_cells), 0

        if(not can_stand_on(start_cell)):
            while(self.is_position_valid(start_cell) and within_distance(start_cell, max_distance)):
                start_cell = self.get_cell_in_direction(cell=start_cell, direction=Direction.DOWN, restrict_boundary=False)
                if(can_stand_on(start_cell)):
                    break
            
            if((not self.is_position_valid(start_cell)) or (not can_stand_on(start_cell))):
                # print(f"Invalid start cell {start_cell}")
                return 0, list(reachable_cells), 0
        
        reachable_cells.add(start_cell)

        # Priority queue for A* search
        open_set = [(0, start_cell)]
        g_score = {start_cell: 0}
        heapq.heapify(open_set)
    
        while open_set:
            _, current = heapq.heappop(open_set)
            x, y = current
            
            for direction, routes in directions_and_routes:
                dx, dy = direction
                nx, ny = (x + dx, y + dy)
                new_cell = (nx, ny)

                if((new_cell == current) or (self.grid[current] == constants.GRID_PLATFORM) or (self.grid[current] == constants.GRID_LAVA)):
                    continue

                if(not (self.is_position_valid(new_cell) and within_distance(new_cell, max_distance))):
                    continue
                
                if(not (can_stand_on(new_cell))):
                   continue

                if(not is_any_route_clear(current, routes)):
                    continue

                # print(f"Current : {current}, New cell : {new_cell}, Direction : {direction}, Route : {routes}")

                if ((new_cell not in reachable_cells)):# or (g_score[new_cell] > (g_score[current] + 1))):
                    g_score[new_cell] = g_score[current] + 1
                    heapq.heappush(open_set, (g_score[new_cell] + manhattan_distance(nx, ny, *start_cell), new_cell))
                    reachable_cells.add(new_cell)

            for direction in [(-2, 2), (-1, 2), (1, 2), (2, 2)]:
                nx, ny = (x + direction[0], y + direction[1])
                new_cell = (nx, ny)

                routes = []
                found = False
                for d, r in directions_and_routes:
                    if(direction == d):
                        routes = r
                        found = True
                        break
                
                if(not found):
                    print("No route found!")
                    break

                if(not is_any_route_clear(current, routes)):
                    continue
                
                # print(f"Current : {current}, direction : {direction}, routes : {routes}")

                if(self.grid[new_cell] != constants.GRID_EMPTY_SPACE):
                    continue

                while(self.is_position_valid(new_cell) and within_distance(new_cell, max_distance)):
                    if(can_stand_on(new_cell)):
                        if ((new_cell not in reachable_cells)):# or (g_score[new_cell] > (g_score[current] + 1))):
                            g_score[new_cell] = g_score[current] + 1
                            heapq.heappush(open_set, (g_score[new_cell] + manhattan_distance(nx, ny, *start_cell), new_cell))
                            reachable_cells.add(new_cell)             
                        break
                    else:
                        new_cell = self.get_cell_in_direction(new_cell, direction=Direction.DOWN, restrict_boundary=False)

                        if((not self.is_position_valid(new_cell)) or (self.grid[new_cell] != constants.GRID_EMPTY_SPACE)):
                            break

        # Find the highest index of a player path cell that is close to any reachable cell
        highest_reached_index = -1
        for reachable_cell in reachable_cells:
                for direction, routes in directions_and_routes:
                    new_cell = (reachable_cell[0] + direction[0], reachable_cell[1] + direction[1])
                    if(new_cell in player_path):
                        if(is_any_route_clear(reachable_cell, routes)):
                            for i, path_cell in enumerate(player_path):
                                if(path_cell == new_cell):
                                    highest_reached_index = max(highest_reached_index, i)

                # if manhattan_distance(reachable_cell[0], reachable_cell[1], path_cell[0], path_cell[1]) <= max_distance:
                #     highest_reached_index = max(highest_reached_index, i)

        # Calculate reachability percentage based on the highest reachable index
        reachability_percentage = (highest_reached_index / (len(player_path) - 1)) if highest_reached_index >= 0 else 0
        return reachability_percentage, list(reachable_cells), highest_reached_index

    def generate_player_path(self, max_turns, randomness):

        grid_width_indices = (6, self.width - 6)
        grid_height_indices = (6, self.height - 6)

        def get_neighbors(x, y, visited):
            neighbors = []
            for dx, dy in [(1, 0), (0, -1), (0, 1)]:  # Left, Right, Up, Down #(-1, 0)
                nx, ny = x + dx, y + dy
                if grid_width_indices[0] <= nx <= grid_width_indices[1] and grid_height_indices[0] <= ny <= grid_height_indices[1] and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            return neighbors
        
        turn_count = 0
        
        path = [self.start_pos]
        visited = set(path)
        current = self.start_pos
        
        while current != self.end_pos:
            # Determine neighbors and choose one based on complexity
            neighbors = get_neighbors(*current, visited)
            
            if not neighbors:
                # Backtrack if no unvisited neighbors (should be rare with proper parameters)
                path.pop()
                if not path:
                    raise ValueError("No path found; consider lower complexity or different start/end.")
                current = path[-1]
                continue
            
            # Select a neighbor based on complexity setting
            if turn_count < max_turns and (random.random() < randomness):
                # High complexity: shuffle and choose any unvisited neighbor
                next_cell = random.choice(neighbors)
                turn_count += 1
            else:
                # Low complexity: Move in direction closer to the goal
                next_cell = min(
                    neighbors, 
                    key=lambda cell: abs(cell[0] - self.end_pos[0]) + abs(cell[1] - self.end_pos[1])
                )

            # Add the chosen cell to the path and mark it as visited
            path.append(next_cell)
            visited.add(next_cell)
            current = next_cell
        
        return path
    
    def calculate_reachability_in_mask(self, path):
        
        def manhattan_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Check proximity for each player path cell
        def within_distance(cell, max_dist):
            return any(manhattan_distance(cell[0], cell[1], px, py) <= max_dist for px, py in path)
        
        def is_any_route_clear(cell, routes):
            is_clear = False
            for route in routes:
                if(are_route_cells_clear(cell, route)):
                    is_clear = True
                    break
            return is_clear
        
        def are_route_cells_clear(cell, directions):
            for (dx, dy) in directions:
                nx, ny = (cell[0] + dx, cell[1] + dy)
                new_cell = (nx, ny)

                if(not self.is_position_valid(new_cell)):
                    return False

                if(not (self.grid[new_cell] == constants.GRID_EMPTY_SPACE)):
                    return False

            return True
  
        def can_stand_on(cell):
            below_cell = self.get_cell_in_direction(cell, direction=Direction.DOWN, restrict_boundary=False)
            if(not self.is_position_valid(below_cell)):
                return False
            return (self.grid[cell] == constants.GRID_EMPTY_SPACE) and (self.grid[below_cell] == constants.GRID_PLATFORM)
        
        max_distance = 5

        reachable_cells = set()

        directions_and_routes = [
            # ((-3, -3), [[]]), 
            
            ((-2, -3), [
                [(0, -1), (0, -2), (0, -3), (-1, -3)]
            ]), 
            
            ((-1, -3), [
                [(0, -1), (0, -2), (0, -3)]
            ]),
            
            # ((0, -3), [[]]), 
            
            ((1, -3), [
                [(0, -1), (0, -2), (0, -3)]
            ]),
            
            ((2, -3), [
                [(0, -1), (0, -2), (0, -3), (1, -3)]
            ]), 
            
            # ((3, -3), [[]]),
            
            # ((-3, -2), [[]]),
            
            ((-2, -2), [
                [(0, -1), (0, -2), (0, -3), (-1, -3), (-2, -3)]
            ]), 
            
            ((-1, -2), [
                [(0, -1), (0, -2)]
            ]), 
            
            # ((0, -2), [[]]), 
            
            
            ((1, -2), [
                [(0, -1), (0, -2)]
            ]), 
            
            ((2, -2), [
                [(0, -1), (0, -2), (0, -3), (1, -3), (2, -3)]
            ]),
            
            # ((3, -2), [[]]),
            # ((-3, -1), [[]]), 
            
            ((-2, -1), [
                [(0, -1), (0, -2), (-1, -2), (-2, -2)]
            ]), 
            
            ((-1, -1), [
                [(0, -1)]
            ]), 
            
            # ((0, -1), [[]]), 
            
            ((1, -1), [
                [(0, -1)]
            ]), 
            
            ((2, -1), [
                [(0, -1), (0, -2), (1, -2), (2, -2)]
            ]), 
            
            # ((3, -1), [[]]),
            

            ((-3,  0), [
                [(0, -1), (-1, -1), (-1, -2), (-2, -2), (-2, -1), (-3, -1)]
            ]), 
            
            ((-2,  0), [
                [(0, -1), (-1, -1), (-2, -1)]
            ]), 
            
            ((-1,  0), [[(-1, 0)]]), 
            
            ((0,  0), [[(0, 0)]]), 
            
            ((1,  0), [[(1, 0)]]), 
            
            ((2,  0), [
                [(0, -1), (1, -1), (2, -1)]
            ]), 
            
            ((3,  0), [
                [(0, -1), (1, -1), (1, -2), (2, -2), (2, -1), (3, -1)]
            ]),

            # ((-3,  1), [[]]), 
            
            ((-2,  1), [
                [(0, -1), (-1, -1), (-2, -1), (-2, 0)]
            ]), 
            
            ((-1,  1), [
                [(-1, 0)]
            ]), 
            
            # ((0,  1), [[]]), 
            
            ((1,  1), [
                [(1, 0)]
            ]),
            
            ((2,  1), [
                [(0, -1), (1, -1), (2, -1), (2, 0)]
            ]), 
            
            # ((3,  1), [[]]),
            # ((-3,  2), [[]]), 
            
            
            ((-2,  2), [
                [(0, -1), (-1, -1), (-2, -1), (-2, 0), (-2, 1)]
            ]),
            
            ((-1,  2), [
                [(-1, 0), (-1, 1)]
            ]),
             
            # ((0,  2), [[]]), 
            
            ((1,  2), [
                [(1, 0), (1, 1)]
            ]), 
            
            ((2,  2), [
                [(0, -1), (1, -1), (2, -1), (2, 0), (2, 1)]
            ]),
            
            # ((3,  2), [[]]),
            # ((-3,  3), [[]]), 
            
            ((-2,  3), [
                [(0, -1), (-1, -1), (-2, -1), (-2, 0), (-2, 1), (-2, 2)]
            ]), 
            
            ((-1,  3), [
                [(-1, 0), (-1, 1), (-1, 2)]
            ]), 
            
            # ((0,  3), [[]]), 
            
            ((1,  3), [
                [(1, 0), (1, 1), (1, 2)]
            ]),
            
            ((2,  3), [
                [(0, -1), (1, -1), (2, -1), (2, 0), (2, 1), (2, 2)]
            ]),
            
            # ((3,  3), [[]]),
            ]

        start_cell = path[0]
        end_cell = path[-1]

        if(self.grid[start_cell] != constants.GRID_EMPTY_SPACE):
            return 0, list(reachable_cells), 0

        if(not can_stand_on(start_cell)):
            while(self.is_position_valid(start_cell) and within_distance(start_cell, max_distance)):
                start_cell = self.get_cell_in_direction(cell=start_cell, direction=Direction.DOWN, restrict_boundary=False)
                if(can_stand_on(start_cell)):
                    break
            
            if((not self.is_position_valid(start_cell)) or (not can_stand_on(start_cell))):
                # print(f"Invalid start cell {start_cell}")
                return 0, list(reachable_cells), 0
        
        reachable_cells.add(start_cell)

        # Priority queue for A* search
        open_set = [(0, start_cell)]
        g_score = {start_cell: 0}
        heapq.heapify(open_set)
    
        while open_set:
            _, current = heapq.heappop(open_set)
            x, y = current
            
            for direction, routes in directions_and_routes:
                dx, dy = direction
                nx, ny = (x + dx, y + dy)
                new_cell = (nx, ny)

                if((new_cell == current) or (self.grid[current] == constants.GRID_PLATFORM) or (self.grid[current] == constants.GRID_LAVA)):
                    continue

                if(not (self.is_position_valid(new_cell) and within_distance(new_cell, 10))):
                    continue
                
                if(not (can_stand_on(new_cell))):
                   continue

                if(not is_any_route_clear(current, routes)):
                    continue

                # print(f"Current : {current}, New cell : {new_cell}, Direction : {direction}, Route : {routes}")

                if ((new_cell not in reachable_cells)):# or (g_score[new_cell] > (g_score[current] + 1))):
                    g_score[new_cell] = g_score[current] + 1
                    heapq.heappush(open_set, (g_score[new_cell] + manhattan_distance(nx, ny, *start_cell), new_cell))
                    reachable_cells.add(new_cell)

            for direction in [(-2, 2), (-1, 2), (1, 2), (2, 2)]:
                nx, ny = (x + direction[0], y + direction[1])
                new_cell = (nx, ny)

                routes = []
                found = False
                for d, r in directions_and_routes:
                    if(direction == d):
                        routes = r
                        found = True
                        break
                
                if(not found):
                    print("No route found!")
                    break

                if(not is_any_route_clear(current, routes)):
                    continue
                
                # print(f"Current : {current}, direction : {direction}, routes : {routes}")

                if(self.grid[new_cell] != constants.GRID_EMPTY_SPACE):
                    continue

                while(self.is_position_valid(new_cell) and within_distance(new_cell, max_distance)):
                    if(can_stand_on(new_cell)):
                        if ((new_cell not in reachable_cells)):# or (g_score[new_cell] > (g_score[current] + 1))):
                            g_score[new_cell] = g_score[current] + 1
                            heapq.heappush(open_set, (g_score[new_cell] + manhattan_distance(nx, ny, *start_cell), new_cell))
                            reachable_cells.add(new_cell)             
                        break
                    else:
                        new_cell = self.get_cell_in_direction(new_cell, direction=Direction.DOWN, restrict_boundary=False)

                        if((not self.is_position_valid(new_cell)) or (self.grid[new_cell] != constants.GRID_EMPTY_SPACE)):
                            break

        # Find the highest index of a player path cell that is close to any reachable cell
        highest_reached_index = -1
        for reachable_cell in reachable_cells:
                for direction, routes in directions_and_routes:
                    new_cell = (reachable_cell[0] + direction[0], reachable_cell[1] + direction[1])
                    if(new_cell in path):
                        if(is_any_route_clear(reachable_cell, routes)):
                            for i, path_cell in enumerate(path):
                                if(path_cell == new_cell):
                                    highest_reached_index = max(highest_reached_index, i)

        # Calculate reachability percentage based on the highest reachable index
        reachability_percentage = (highest_reached_index / (len(path) - 1)) if highest_reached_index >= 0 else 0
        return reachability_percentage, list(reachable_cells), highest_reached_index

    def find_hanging_cells(self):
        """
        Find all cells in the grid that are hanging (not connected to the starting cell via empty spaces).
        
        :param grid: 2D numpy array, where 0 represents empty spaces, and other values are obstacles.
        :param start_cell: Tuple (x, y), the starting cell (column, row).
        :return: List of tuples representing the coordinates of all hanging cells.
        """

        grid = self.grid
        start_cell = self.start_pos

        width, height = grid.shape
        x_start, y_start = start_cell

        if(not self.is_position_valid(start_cell)):
            return []

        # Create a visited array to track reachable cells
        visited = np.zeros_like(grid, dtype=bool)
        
        # Directions for 4-connectivity (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # BFS/DFS to mark reachable cells
        def flood_fill(x, y):
            stack = [(x, y)]
            visited[x, y] = True
            while stack:
                cx, cy = stack.pop()
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy

                    if(not self.is_position_valid((nx, ny))):
                       continue

                    if grid[nx, ny] == constants.GRID_EMPTY_SPACE and not visited[nx, ny]:  # Check empty and unvisited
                        visited[nx, ny] = True
                        stack.append((nx, ny))

        # Start flood fill from the starting cell
        flood_fill(x_start, y_start)

        # Find all hanging cells: Unvisited empty cells
        hanging_cells = [(x, y) for x in range(width) for y in range(height) if grid[x, y] == constants.GRID_EMPTY_SPACE and not visited[x, y]]

        return hanging_cells
    
    def is_cell_hanging(self, cell):

        if(self.grid[cell] != constants.GRID_EMPTY_SPACE):
            return False

        is_hanging = True

        for dir in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            next_cell = self.get_cell_in_direction(cell, dir, restrict_boundary=False)
            if(not self.is_position_valid(next_cell)):
                continue

            if(self.grid[next_cell] == constants.GRID_EMPTY_SPACE):
                is_hanging = False
                break

        return is_hanging

            
    def get_cells_in_action_mask_region(self):

        cells_in_mask = []

        posX, posY = self.player_pos
        mask_to_draw = self.mask
        pivotX, pivotY = (math.floor(mask_to_draw.shape[0] / 2), math.floor(mask_to_draw.shape[1] / 2))
        for local_X in range(0, mask_to_draw.shape[0]):
            for local_Y in range(0, mask_to_draw.shape[1]):
                x, y = posX + local_X - pivotX, posY + local_Y - pivotY
                cells_in_mask.append((x, y))

        return cells_in_mask

    def save_screen_image(self, full_path):
        pygame.image.save(self.display, full_path)
        print(f"Saved image : {full_path}")