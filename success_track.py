# Keeps track of how many runs out of 10 finished for each task
# Produces a table
# Dir: 'results/{task_id}/Rep_{slurm_array_id}/{task_id}-{slurm_id}/results.csv'
# csv structure: task_id,seed,best_cv_score,test_score

import pandas as pd
import os
import re

# TODO: when you go into task_id folder, check if there are 10 folders inside


if __name__ == "__main__":
    results_path = 'results/'

    df = pd.read_csv('/common/suzuek/ea-tpe-random/data/task_list.csv')
    df = df.sort_values(by='rows', ascending=True)
    # get task_id column as list
    task_ids = df['task_id'].tolist()

    summary = [] # stores (task_id, completed runs)
    for task_id in task_ids:
        task_dir = os.path.join(results_path, str(task_id))
        if not os.path.exists(task_dir):
            summary.append((task_id, 0))
            continue
        
        # count subfolders that contain a valid results.csv
        completed = 0
        for rep_folder in os.listdir(task_dir):
            # {task_id}/Rep_{slurm_id}
            rep_path = os.path.join(task_dir, rep_folder)
            if not os.path.isdir(rep_path): 
                print(f"{rep_path} doesn't exist.")
                continue
            
            # {task_id}/Rep_{slurm_id}/{task_id}-{slurm_id}/results.csv
            for sub in os.listdir(rep_path):
                sub_path = os.path.join(rep_path, sub)
                results_csv_path = os.path.join(sub_path, "results.csv")
                if os.path.exists(results_csv_path): completed += 1
        
        summary.append((task_id, completed))

    summary_df = pd.DataFrame(summary, columns=["task_id", "completed_runs"])
    summary_df["completion_rate"] = summary_df["completed_runs"] / 10

    summary_df.to_csv("completion_summary.csv", index=False)