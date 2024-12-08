
import os
import json
import numpy as np

def export_level(env, base_directory, level_id):
    
    env.save_screen_image(os.path.join(base_directory, f"level_{level_id}_img.png"))
    with open(os.path.join(base_directory, f'level_{level_id}_path'), 'wb') as fp:
        np.save(fp, env.player_path)
    with open(os.path.join(base_directory, f'level_{level_id}_grid'), 'wb') as fp:
        np.save(fp, env.grid)
    with open(os.path.join(base_directory, f'level_{level_id}_data.json'), 'wb') as fp:
        json.dump(env.get_exportable_format(), os.path.join(base_directory, f'level_{level_id}_data.json'))