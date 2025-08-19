"""
Script to filter out datasets with fewer than 15 runs.
"""

import os
import pandas as pd

results_dir = './results/'
max_runs = 15

complete_datasets: list = []

# loop through each item in results dir
for item in os.listdir(results_dir):

    # './results/108', or ./results/text.txt for example
    full_path = os.path.join(results_dir, item)

    # if item is a directory, and its name is a numeric id (a folder containing dataset results)
    if os.path.isdir(full_path) and item.isdigit(): 
        print("Currently in folder: ", full_path)

        # count number of complete runs! (there should be 15 of them) 
        run_count = 0
        for sub_item in os.listdir(full_path): 

            # e.g. './results/108/Rep_1
            sub_path = os.path.join(full_path, sub_item)

            # identify folders that start with "Rep_"
            if os.path.isdir(sub_path) and sub_item.startswith("Rep_"):
                
                # additional check to see if "Rep_" folders contain 1 inner folder, which contains a "results.csv" file
                entries = [e for e in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, e))]
                if len(entries) == 1:
                    # e.g. './results/108/Rep_1/1-108/results.csv'
                    results_path = os.path.join(sub_path, entries[0],'results.csv')
                    if os.path.exists(results_path) and len(pd.read_csv(results_path)) == 1:
                        print(sub_item)
                        run_count += 1

        assert run_count <= max_runs, f"uh oh! over {max_runs} runs? how is this possible?"
        if run_count == max_runs: complete_datasets.append(item)

print(f"Datasets with {max_runs} runs: ", complete_datasets)

# Assuming agg_test.csv is in results dir
agg_test_dir = os.path.join(results_dir, 'agg_test.csv')
if (os.path.isfile(agg_test_dir)):
    old_df = pd.read_csv(agg_test_dir)
    old_df['task_id'] = old_df['task_id'].astype(str)
    filtered_df = old_df[old_df['task_id'].isin(complete_datasets)]

    print(filtered_df)


    filtered_df.to_csv('filtered_scores.csv', index=False)