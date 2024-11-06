import pygame
import numpy as np
import math
import constants

import heapq

from constants import Direction
from collections import deque
from typing import Optional

import gymnasium as gym

class GameWorld(gym.Env):

    def __init__(self, width, height, player_path, mask_size = (3, 3), num_tile_actions = 3):

        self.width = width
        self.height = height

        self.screen_resolution_X, self.screen_resolution_Y = 1024, 768
        self.screen_resolution = (self.screen_resolution_X, self.screen_resolution_Y)

        self.display = pygame.display.set_mode(self.screen_resolution)
        pygame.display.set_caption("Game")
        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self.grid = np.zeros([self.width, self.height], np.uint8)

        self.start_pos = (1, self.height // 2)
        self.end_pos = (self.width - 2, self.height // 2)

        self.player_pos = self.start_pos
        self.coverable_path = []

        path_list = [self.start_pos]
        self.mask = np.full(mask_size, 0, dtype=np.uint8)

        self.player_path = path_list
        self.max_frame_count = 1000
        self.iterations_per_game = 1

        self.reset_count = 0

        self.mask_size = mask_size
        self.num_tile_actions = num_tile_actions

        # Action space: Each element in the 2D mask has 3 possible values (0, 1, or 2)
        self.action_space = gym.spaces.MultiDiscrete([num_tile_actions] * (self.mask_size[0] * self.mask_size[1]))

        # Observation space: (4 channels, grid_size X, grid_size Y)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.width, self.height, 4), dtype=np.uint8
        )

        self.set_player_path(player_path)
        self.reset()

    
    def _get_obs(self):
      
        normalized_grid = self.grid.copy()
        normalized_grid = np.round((normalized_grid / (constants.TOTAL_NUMBER_OF_TILE_TYPES - 1)) * 255).astype(np.uint8)

        ohe_grid_player_path = np.zeros_like(self.grid, dtype=np.uint8)        
        for cell in self.player_path:
            ohe_grid_player_path[cell] = 1
        
        ohe_grid_player_pos = np.zeros_like(self.grid, dtype=np.uint8)
        ohe_grid_player_pos[self.player_pos] = 1

        window_size = ((5, 5))
        window = np.full(window_size, 0, dtype = np.uint8)
        window_grid = np.full_like(self.grid, 0, dtype = np.uint8)

        base_pos = (self.player_pos[0] - (math.floor(window_size[0] / 2)), self.player_pos[1] - math.floor(window_size[1] / 2))
        for x in range(0, window_size[0]):
            for y in range(0, window_size[1]):
                world_pos = base_pos[0] + x, base_pos[1] + y
                if(self.is_position_valid(world_pos)):
                    window[x, y] = self.grid[world_pos]
                    window_grid[world_pos] = self.grid[world_pos]
   
        state = np.stack([normalized_grid, ohe_grid_player_path * 255, ohe_grid_player_pos * 255, window_grid * 255], axis=0)
        return state.transpose(1, 2, 0) # Shape to (grid_size X, grid_size Y, 4 channels)

        # window_size = (11, 11)
        # window_grid = np.full(window_size, -1, dtype = np.int32)
        # window_player_path = np.full(window_size, 0, dtype = np.int32)
        # window_player_pos = np.full(window_size, 0, dtype = np.int32)

        # base_pos = (self.player_pos[0] - (math.floor(window_size[0] / 2)), self.player_pos[1] - math.floor(window_size[1] / 2))
        # for x in range(0, window_size[0]):
        #     for y in range(0, window_size[1]):
        #         world_pos = base_pos[0] + x, base_pos[1] + y
        #         if(self.is_position_valid(world_pos)):
        #             window_grid[x, y] = self.grid[world_pos]
        #         if(world_pos in self.player_path):
        #             window_player_path[x, y] = 1 
        #         if(world_pos == self.player_pos):
        #             window_player_pos[x, y] = 1       
   
        # state = np.stack([window_player_path, window_player_pos, window_grid], axis=0)

        return state
    
    def _get_info(self):
        return {
            "progress": (self.player_path_index / len(self.player_path))
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        
        self.frame_count = 0
        self.grid = np.full([self.width, self.height], constants.GRID_PLATFORM, np.uint8)
        self.player_pos = self.start_pos
        self.player_path_index = 0

        self.reset_count += 1
        
        # print(f"Game reset : {self.reset_count}")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        # Convert the flat action back to a 2D action mask
        action_mask = np.array(action).reshape((self.mask_size[0], self.mask_size[1]))

        # print(action_mask_2d.shape)
        # print(action_mask_2d)

        pygame.event.get()
        
        self._update(False)
        reward_before_action = self.get_current_reward()

        # Deciding action
        self.mask = action_mask
        self.execute_tile_mask(self.mask)
        
        state = self._get_obs()
        reward_after_action = self.get_current_reward()
        reward = (reward_after_action - reward_before_action)

        # print(f"Reward, B: {reward_before_action} A: {reward_after_action}")

        terminated = self.frame_count > self.max_frame_count
        truncated = False

        self.player_path_index = (self.player_path_index + 1) % len(self.player_path)
        self.player_pos = self.player_path[self.player_path_index]

        self.render(True)
        self.frame_count += 1

        return state, reward, terminated, truncated, self._get_info() 

    def get_current_reward(self):

        if(len(self.player_path) == 0):
            print("PLAYER PATH IS EMPTY!!")
            reward = -10000
            return reward

        reward, self.coverable_path = self.calculate_reachability(max_distance=6)

        reward *= 1000

        # Checking if lava tiles are surrounded with proper cells
        for x in range(0, self.width):
            for y in range(0, self.height):
                cell = (x, y)
                if(self.grid[cell] == constants.GRID_LAVA):
                    for d in [Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                        neighbor_cell = self.get_cell_in_direction(cell, direction=d,restrict_boundary=False)
                        if(not self.is_position_valid(neighbor_cell)):
                            continue
                        if(self.grid[neighbor_cell] == constants.GRID_EMPTY_SPACE):
                            reward -= 1
                            break

        lava_tile_count = (self.grid == constants.GRID_LAVA).sum()
        reward = reward - (min(0, lava_tile_count - 10) * 0.5)

        # for x in range(0, self.width):
        #     for y in range(0, self.height):
        #         if((x, y) in self.player_path):
        #             continue
        #         if(self.grid[x, y] == constants.GRID_PLATFORM):
        #             reward += 1
            
        # reward /= ((self.width * self.height) - len(self.player_path))
        # reward *= 2

        blocks = 0
        for cell in self.player_path:
            if(self.grid[cell] == constants.GRID_PLATFORM):
                blocks += 1

        reward += (2 * (1.0 - (blocks / len(self.player_path))))

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
        self.max_frame_count = (len(self.player_path)) * self.iterations_per_game

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

        pygame.draw.rect(self.display, constants.COLOR_CYAN, rect= pygame.Rect(self.player_pos[0] * cell_draw_size, self.player_pos[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 1, border_radius = 1)
        pygame.draw.rect(self.display, constants.COLOR_GREEN, rect= pygame.Rect(self.start_pos[0] * cell_draw_size, self.start_pos[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 1)
        pygame.draw.rect(self.display, constants.COLOR_PURPLE, rect= pygame.Rect(self.end_pos[0] * cell_draw_size, self.end_pos[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 1)

        # Render ghost kernel
        posX, posY = self.player_pos
        cell_draw_size = constants.CELL_DRAW_SIZE
        mask_to_draw = self.mask
        pivotX, pivotY = (math.floor(mask_to_draw.shape[0] / 2), math.floor(mask_to_draw.shape[0] / 2))
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

                

        for cell in self.player_path:
            pygame.draw.rect(self.display, constants.COLOR_CYAN, rect= pygame.Rect(cell[0] * cell_draw_size, cell[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= 1, border_radius = 0)

        for cell in self.coverable_path:
            pygame.draw.rect(self.display, constants.COLOR_MAGENTA, rect= pygame.Rect(cell[0] * cell_draw_size, cell[1] * cell_draw_size, cell_draw_size, cell_draw_size), width= cell_draw_size, border_radius = 0)

        if(flip_display):
            pygame.display.flip()

    def close(self):
        #print("Closing game")    
        pass

    def calculate_reachability(self, max_distance=3):

        def manhattan_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)
        
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
        
        grid = self.grid
        player_path = self.player_path

        height, width = grid.shape
        reachable_cells = set()
        reachable_path_cells = set()  # Cells near the player path that can be reached
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

        # Priority queue for A* search
        open_set = [(0, start_cell)]
        g_score = {start_cell: 0}
        heapq.heapify(open_set)
        
        # Check proximity for each player path cell
        def within_distance(cell, max_dist):
            return any(manhattan_distance(cell[0], cell[1], px, py) <= max_dist for px, py in player_path)

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
        return reachability_percentage, list(reachable_cells)

    def save_screen_image(self, full_path):
        pygame.image.save(self.display, full_path)
        print(f"Saved image : {full_path}")