import numpy as np
import json
import os
import game_world
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if(__name__ == "__main__"):

    paths_data_file_path = "./saves/paths_data.json"
    paths_data = []
    with open(paths_data_file_path, "r") as file:
        paths_data = json.load(file)

    results_tuples = [
        ("Baseline - Random Level Generator", "./saves/evaluation/approach_1/results.json"),
        ("Reachability-guided Random Level Generator", "./saves/evaluation/approach_2/results.json"),
        ("Recurrent PPO (Proposed approach)", "./saves/evaluation/proposed_approach/results.json")
    ]

    for approach_name, results_path in results_tuples:

        results = {}
        with open(results_path, "r") as file:
            results = json.load(file)

        solvable_count = 0

        complexities = []
        reachabilities = []
        hanging_cell_counts = []
        solvabilities = []

        base_directory = f"./saves/visualizations/{approach_name}"
        os.makedirs(base_directory, exist_ok=True)

        for r in results:

            path_id = r["path_id"]
            path = r["path_data"]["path"]
            path_complexity = r["path_data"]["complexity"]
            is_solvable = r["env_data"]["is_solvable"]
            hanging_cell_count = len(r["env_data"]["hanging_cells"])
            reachability = r["env_data"]["reachability"]

            complexities.append(path_complexity)
            reachabilities.append(reachability)
            hanging_cell_counts.append(hanging_cell_count)
            solvabilities.append(is_solvable)

            for index, cell in enumerate(path):
                    path[index] = (cell[0], cell[1])

            if(is_solvable):
                
                solvable_count += 1

                # env_data = r["env_data"]
                # grid = env_data["grid"]
                # env.grid = np.array(grid)
                # env.render(flip_display=True)
                # pygame.image.save(env.display, os.path.join("./saves/visualizations", f"test_img.png"))
                
        print(f"Solvable count : {solvable_count} / {len(results)}" )

        solvable_color = (89/255, 161/255, 79/255)
        unsolvable_color = (225/255, 86/255, 89/255)

        solvable_patch = mpatches.Patch(color=solvable_color, label='Solvable Level')
        unsolvable_patch = mpatches.Patch(color=unsolvable_color, label='Unsolvable Level')

        x = np.arange(len(complexities))  # X-axis positions
        # Map solvabilities to colors (True -> green, False -> red)
        colors = [solvable_color if solvable else unsolvable_color for solvable in solvabilities]



        # Set up the bar width and figure size
        bar_width = 0.75
        fig, ax = plt.subplots(figsize=(24, 8))
        ax.bar(x, complexities, width=bar_width, color = colors)
        # Add labels and legend
        ax.set_xlabel('Level', fontsize=14)
        ax.set_ylabel('Path Complexity', fontsize=14)
        ax.set_title(f'Path Complexities in Levels - {approach_name}', fontsize=16)
        plt.legend(handles=[solvable_patch, unsolvable_patch], loc='upper left')
        ax.legend(fontsize=12)
        ax.grid(True)


        # Save the plot locally
        plt.tight_layout()
        plt.savefig(os.path.join(base_directory, 'path_complexities.png'))
        plt.close()

        ####################################################

        # Set up the bar width and figure size
        bar_width = 0.75
        fig, ax = plt.subplots(figsize=(24, 8))
        ax.bar(x, reachabilities, width=bar_width, color = colors)
        # Add labels and legend
        ax.set_xlabel('Level', fontsize=14)
        ax.set_ylabel('Reachability', fontsize=14)
        ax.set_title(f'Reachabilities of Levels - {approach_name}', fontsize=16)
        plt.legend(handles=[solvable_patch, unsolvable_patch], loc='upper left')
        ax.legend(fontsize=12)
        ax.grid(True)

        # Save the plot locally
        plt.tight_layout()
        plt.savefig(os.path.join(base_directory, 'reachabilities.png'))
        plt.close()

        ####################################################

        # Set up the bar width and figure size
        bar_width = 0.75
        fig, ax = plt.subplots(figsize=(24, 8))
        ax.bar(x, hanging_cell_counts, width=bar_width, color = colors)
        # Add labels and legend
        ax.set_xlabel('Level', fontsize=14)
        ax.set_ylabel('Number of Hanging cells in Level', fontsize=14)
        ax.set_title(f'Hanging cell counts - {approach_name}', fontsize=16)
        plt.legend(handles=[solvable_patch, unsolvable_patch], loc='upper right')
        ax.legend(fontsize=12)
        ax.grid(True)

        # Save the plot locally
        plt.tight_layout()
        plt.savefig(os.path.join(base_directory, 'hanging_cell_counts.png'))
        plt.close()

        print(f"Plots saved for approach {approach_name}")
