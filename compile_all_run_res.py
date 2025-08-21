"""
Script to compile run-by-run results from Random, EA, TPEBO, TPEC RF experiments.
Produces "all_run_scores.csv".
Requires the following:
- 'filtered_scores.csv' (contains task IDs with exactly 15 runs), 
- 'results' folder (completed Random runs)
- 'rfresults' folder (completed EA, TPEBO, and TPEC runs)

(Sorry for the confusing naming conventions!)

Workflow:
1. aggregate_random_res.py
    Computes average test scores for Random, 
    needed to identify the most challenging tasks (<80% accuracy)
    Output: random_scores.csv

2. filter_datasets.py
    Filters out tasks that have fewer than 15 runs
    Output: filtered_task_list.csv, filtered_scores.csv

3. compile_all_run_res.py (this file! :D)
"""
import pandas as pd
import os
import re

# Path structure (random runs): results/{task_id}/
#                               Rep_{SLURM_ARRAY_TASK_ID}/
#                               {SLURM_ARRAY_TASK_ID}-{task_id}/
#                               results.csv

# Path structure (non-random experiments): './rfresults/{TASK ID}/result_{METHOD}_{RUN}.jsonl

# all_run_scores.csv structure:
# task_id | method | run | test_accuracy_score | cv_accuracy_score

def make_random_df(task_ids, results_path):
    reps_by_task: dict = {} # id: [SLURM array task IDs]
    for id in task_ids:
        id_random_path = os.path.join(results_path, str(id))

        # it should exist, but just in case
        if not os.path.isdir(id_random_path): 
            print(f"Task {id} does not exist.")
        else:
            # store names (strings) of all 'Rep_{SLURM_ARRAY_TASK_ID}' folders
            rep_folders = os.listdir(id_random_path)

            # extract SLURM array ID for current task
            rep_numbers = []
            for folder in rep_folders:
                match = re.search(r'Rep_(\d+)', folder)
                if match:
                    rep_numbers.append(int(match.group(1)))

            reps_by_task[id] = rep_numbers


    cv_scores_by_task: dict = {} # id: [score, score, score...]
    test_scores_by_task: dict = {} # id: [score, score, score...]
    for task, replicates in reps_by_task.items():
        test_scores = []
        cv_scores = []
        # Retrieve test scores from each replicate
        for rep in replicates:
            results_path = os.path.join(results_path, f"{task}/Rep_{rep}/{rep}-{task}/results.csv")
            # If folder exists, but not results.csv
            if not os.path.exists(results_path):
                print(f"Missing results.csv for task {task}, replicate {rep}, skipping.")
                continue

            df = pd.read_csv(results_path)
            test_scores.append(df['test_score'].iloc[0])
            cv_scores.append(df['best_cv_score'].iloc[0])

        test_scores_by_task[task] = test_scores
        cv_scores_by_task[task] = cv_scores
    
    assert(test_scores_by_task.keys() == cv_scores_by_task.keys())

    # random_df = pd.DataFrame( {
    #     'task_id': list()
    # })


    
    # return random_df


if __name__ == "__main__":
    # we did 15 reps for random, and 20 for the rest
    non_random_run_count = 20 
    random_run_Count = 15

    filtered_random_scores_path = 'results/filtered_random_scores.csv' 
    random_results_path = 'results/'
    other_results_path = 'rfresults/' # this directory stores results from EA, TPEBO, and TPEC
 
    # contains scores of task IDs with exactly 15 runs
    filtered_random_scores_df = pd.read_csv(filtered_random_scores_path)

    # Retrieve names of hardest tasks (less than 80% accuracy on test set)
    filtered_random_scores_hard_df = filtered_random_scores_df[filtered_random_scores_df['avg_test_score'] < 0.8]
    filtered_task_ids = filtered_random_scores_hard_df['task_id'].tolist()
    filtered_task_ids.sort()

    random_df = make_random_df(filtered_task_ids, random_results_path)
    # ea_df, tpebo_df, tpec_df = make_other_dfs(filtered_task_ids, other_results_path)

