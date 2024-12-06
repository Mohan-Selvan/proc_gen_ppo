import numpy as np
import json

def export_env(env, complete_file_path):
    data = { 
        "grid_size" : (env.width, env.height),
        "grid" : env.grid.tolist(),
        "player_path" : env.player_path
        }
    write_dictionary_to_file(data, complete_file_path)

def write_dictionary_to_file(data, complete_file_path):
    print(f"Writing data to file : {complete_file_path}")
    with open(complete_file_path, "w") as write_file:
        json.dump(data, write_file)
    
    print("File written successfully!")