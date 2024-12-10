import numpy as np
import json
import os
import game_world
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if(__name__ == "__main__"):

    results_app_1 = {}
    with open("./saves/evaluation/approach_2/results.json", "r") as file:
        results_app_1 = json.load(file)

    results_app_2 = {}
    with open("./saves/evaluation/proposed_approach/results.json", "r") as file:
        results_app_2 = json.load(file)

    solvable_levels_in_app_1 = []

    hanging_cell_counts_app_1 = []
    for index, r in enumerate(results_app_1):

        if(r["env_data"]["is_solvable"]):
            hanging_cell_count = len(r["env_data"]["hanging_cells"])
            hanging_cell_counts_app_1.append(hanging_cell_count)
            solvable_levels_in_app_1.append(index)

    hanging_cell_counts_app_2 = []
    for index, r in enumerate(results_app_2):
        if(index in solvable_levels_in_app_1):
            hanging_cell_count = len(r["env_data"]["hanging_cells"])
            hanging_cell_counts_app_2.append(hanging_cell_count)

    base_directory = f"./saves/visualizations/"
    os.makedirs(base_directory, exist_ok=True)

    x = np.arange(1, len(hanging_cell_counts_app_1) + 1)  # X-axis positions

    # Bar width
    bar_width = 0.25

    # X positions for bars
    x1 = np.arange(len(hanging_cell_counts_app_1))
    x2 = x1 + bar_width

    plt.figure(figsize=(10, 6))

    # Plotting the bars
    plt.bar(x1, hanging_cell_counts_app_1, width=bar_width, label="Reachability-guided Random generator", color='goldenrod')
    plt.bar(x2, hanging_cell_counts_app_2, width=bar_width, label="Recurrent PPO model (Proposed approach)", color='royalblue')

    # Adding labels and title
    plt.xlabel("Levels solved by both Reachability-guided Random generator and Recurrent PPO model")
    plt.ylabel("Number of Hanging cells")
    plt.title("Comparison of Hanging Cell Count")
    plt.xticks(x1 + bar_width / 2, [str(i) for i in x])  # Center ticks between bars
    plt.yticks(np.arange(0, 21, 1))
    plt.ylim(0, 20) 
    plt.legend()
    plt.grid(True)

    # Save the plot locally
    plt.tight_layout()
    plt.savefig(os.path.join(base_directory, 'hc_comparison.png'))
    plt.close()

