import pandas as pd
from pathlib import Path
import os
import re
import numpy as np


# Path structure: results/{task_id}/
#                 Rep_{SLURM_ARRAY_TASK_ID}/
#                 {SLURM_ARRAY_TASK_ID}-{task_id}/
#                 results.csv

# Fetch a list of task IDs
# task_id_df = pd.read_csv('/common/suzuek/ea-tpe-random/data/task_list.csv')
task_id_df = pd.read_csv('data/task_list.csv') # FOR TESTING ONLY
task_ids = task_id_df['task_id'].tolist()

# For each task_id, store a list of replicate numbers like so:
# task_id: [1, 2, 3, ..., 15]
reps_by_task = {}
# res_dir = '/common/suzuek/ea-tpe-random/results/'
res_dir = 'results/' # FOR TESTING ONLY
for task_id in task_ids:
    path = os.path.join(res_dir, str(task_id))
    if not os.path.isdir(path):
        print(f"Task {task_id} does not exist.")
    else:
        # store names (strings) of all 'Rep_{SLURM_ARRAY_TASK_ID}' folders
        rep_folders = os.listdir(path)

        # extract SLURM array ID for current task
        rep_numbers = []
        for folder in rep_folders:
            match = re.search(r'Rep_(\d+)', folder)
            if match:
                rep_numbers.append(int(match.group(1)))

        reps_by_task[task_id] = rep_numbers


# Store a list of test scores for each task_id
test_scores_by_task = {}
for task, replicates in reps_by_task.items():
    scores = []
    # Retrieve test scores from each replicate
    for rep in replicates:
        results_path = os.path.join(res_dir, f"{task}/Rep_{rep}/{rep}-{task}/results.csv")
        # If folder exists, but not results.csv
        if not os.path.exists(results_path):
            print(f"Missing results.csv for task {task}, replicate {rep}, skipping.")
            continue

        df = pd.read_csv(results_path)
        scores.append(df['test_score'].iloc[0])
    test_scores_by_task[task] = scores


avg_test_score_by_task = {}
std_test_score_by_task = {}
for task, test_scores in test_scores_by_task.items():
    if len(test_scores) == 0:
        print(f"No valid scores found for task {task}, skipping.")
        continue
    avg_test_score_by_task[task] = np.mean(test_scores)
    std_test_score_by_task[task] = np.std(test_scores)

print(avg_test_score_by_task, std_test_score_by_task)

# Make dataframe
agg_df = pd.DataFrame({
    'task_id': list(avg_test_score_by_task.keys()),
    'avg_test_score': list(avg_test_score_by_task.values()),
    'std_test_score': list(std_test_score_by_task.values()),
})

# Sort by average (ascending)
agg_df = agg_df.sort_values(by='avg_test_score', ascending=True)
agg_df.to_csv(os.path.join(res_dir, "agg_test.csv"), index=False)

