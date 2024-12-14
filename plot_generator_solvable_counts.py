import numpy as np
import json
import os
import game_world
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if(__name__ == "__main__"):

    results_app_1 = {}
    with open("./saves/evaluation/approach_1/results.json", "r") as file:
        results_app_1 = json.load(file)

    results_app_2 = {}
    with open("./saves/evaluation/approach_2/results.json", "r") as file:
        results_app_2 = json.load(file)

    results_app_3 = {}
    with open("./saves/evaluation/proposed_approach/results.json", "r") as file:
        results_app_3 = json.load(file)

    approaches = ["Random generator (Baseline)", "Reachability guided Random generator", "Recurrent PPO (Proposed method)"]
    solvable_counts = []

    for results in [results_app_1, results_app_2, results_app_3]:
         
        solvable_count = 0

        for r in results["levels"]:
             if(r["env_data"]["is_solvable"]):
                  solvable_count += 1

        solvable_counts.append(solvable_count)  

    base_directory = f"./saves/visualizations/"
    os.makedirs(base_directory, exist_ok=True)

    # Define colors for each approach
    colors = plt.cm.tab10(range(len(approaches)))  # Use a colormap for distinct colors

    bar_width = 0.75

    # X-axis positions
    x = np.arange(len(solvable_counts))

    # Create the bar plot
    plt.figure(figsize=(6, 6))
    bars = []
    for i in range(len(approaches)):
        bars.append(plt.bar(x[i], solvable_counts[i], color=colors[i], width=bar_width, label=approaches[i]))

    # Add labels, title, and grid
    plt.xlabel("Approaches")
    plt.ylabel("Solvability Rate")
    plt.ylim((0, 100))
    plt.title("Comparison of Solvable Levels by Different Approaches")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the legend
    plt.legend(title="Approaches", loc='upper left')

    # X-axis ticks
    plt.xticks(x, [str((i + 1)) for i in x])

    # Display the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(base_directory, 'solvable_count_comparison.png'))
    plt.close()

