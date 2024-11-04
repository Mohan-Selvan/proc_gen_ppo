from queue import PriorityQueue
from constants import Direction
import constants
import numpy as np
import math

def heuristic_cost(node1, node2): # Manhattan distance
    x1, y1 = node1
    x2, y2 = node2
    return (abs(x1 - x2) - abs(y1 - y2))

def get_cell_in_direction(cell, direction):
    x, y = cell
    if(direction == Direction.UP):
        y -= 1
    elif(direction == Direction.RIGHT):
        x += 1
    elif(direction == Direction.DOWN):
        y += 1
    elif(direction == Direction.LEFT):
        x -= 1        
    return (x, y)

def find_path(grid, from_node, to_node):

    width = grid.shape[0]
    height = grid.shape[1]

    def is_position_valid(position):
        x, y = position
        return x >= 0 and x < width and y >= 0 and y < height

    path = {}
    scores = {}

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            scores[(x, y)] = {'g': float('inf'), 'f': float('inf')}

    scores[from_node]['g'] = 0
    scores[from_node]['h'] = heuristic_cost(from_node, to_node)

    pq = PriorityQueue()
    pq.put((heuristic_cost(from_node, to_node) + 0, heuristic_cost(from_node, to_node), from_node)) # Tuple format -> (f_score, h_score, cell)

    while not pq.empty():
        current_cell = pq.get()[2] # Getting cell tuple
        
        # Exit if current_cell is the destination
        if(current_cell == to_node):
            break

        # Explore neighbors
        for d in constants.DIRECTIONS:
            
            child_cell = get_cell_in_direction(current_cell, d)

            # If cell is invalid, continue
            if(not is_position_valid(child_cell)):
                continue
                
            # If path is blocked, continue
            if(grid[child_cell] == constants.GRID_PLATFORM):
                print("Platform encountered")
                continue

            g_score = scores[current_cell]['g'] + 1
            f_score = g_score + heuristic_cost(child_cell, to_node)

            if(f_score < scores[child_cell]['f']):
                scores[child_cell]['g'] = g_score
                scores[child_cell]['f'] = f_score
                pq.put((f_score, heuristic_cost(child_cell, to_node), child_cell))
                path[child_cell] = current_cell
    
    result_path = {}
    cell = to_node

    while(cell != from_node):
        result_path[path[cell]] = cell
        cell = path[cell]


    # Formatting path from a dictionary to list
    path_list = []
    path_list.append(from_node)
    if(len(result_path) > 0):
        node = path_list[0]
        while(node in result_path):
            node = result_path[node]
            path_list.append(node)

    return path_list
