"""
Script to compile run-by-run results from Random, EA, TPEBO, TPEC RF experiments.
Produces "all_method_scores_by_run.csv".
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
from natsort import natsorted
import pandas as pd
import os
import re
import json

# Path structure (random runs): results/{task_id}/
#                               Rep_{SLURM_ARRAY_TASK_ID}/
#                               {SLURM_ARRAY_TASK_ID}-{task_id}/
#                               results.csv

# Path structure (non-random experiments): './rfresults/{TASK ID}/result_{METHOD}_{RUN}.jsonl

# all_run_scores.csv structure:
# task_id | method | run | test_accuracy_score | cv_accuracy_score

non_random_run_count = 20 
random_run_count = 15


# function to parse a single file of multiple JSON objects for test score and cv score
def parse_multiple_json_for_score(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split at '}{' but keep the braces using a positive lookahead/lookbehind
    json_objects = re.split(r'(?<=\})\s*(?=\{)', content)

    test_scores = []
    cv_scores = []
    for i, obj in enumerate(json_objects):
        try:
            # Fix "False"/"True" stringified booleans if needed
            obj_fixed = re.sub(r'"\s*(True|False)\s*"', lambda m: m.group(1).lower(), obj)

            data = json.loads(obj_fixed)
            test_scores.append(data["test_accuracy_score"])
            cv_scores.append(-data["cv_accuracy_score"])
        except Exception as e:
            print(f"Error parsing object #{i+1}: {e}")
            continue

    return test_scores, cv_scores


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
            rep_path = os.path.join(results_path, f"{task}/Rep_{rep}/{rep}-{task}/results.csv")
            # If folder exists, but not results.csv
            if not os.path.exists(rep_path):
                print(f"Missing results.csv for task {task}, replicate {rep}, skipping.")
                continue

            df = pd.read_csv(rep_path)
            test_scores.append(df['test_score'].iloc[0])
            cv_scores.append(df['best_cv_score'].iloc[0])

        test_scores_by_task[task] = test_scores
        cv_scores_by_task[task] = cv_scores
    
    assert(test_scores_by_task.keys() == cv_scores_by_task.keys())

    records = []
    for task, test_scores in test_scores_by_task.items():
        cv_scores = cv_scores_by_task[task]
        for run, (test, cv) in enumerate(zip(test_scores, cv_scores)):
            records.append({
                "task_id": task,
                "run": run,
                "test_score": test,
                "cv_score": cv
            })
    
    random_df = pd.DataFrame(records)
    return random_df


def make_other_dfs(task_ids, results_path):
    # Collect paths to JSON files for each method 
    ea_json_names: dict[list] = {} # id: [filenames...]
    tpebo_json_names: dict[list] = {}
    tpec_json_names: dict[list] = {}

    # Iterate through folders in 'rfresults'
    for id in task_ids:
        id_folder = os.path.join(results_path, str(id))
        if not os.path.isdir(id_folder): continue # if a specific id does not exist in 'rfresults', skip

        ea_json_names[id] = [os.path.join(id_folder, filename) for filename in os.listdir(id_folder) if filename.startswith("result_EA")]
        tpebo_json_names[id] = [os.path.join(id_folder, filename) for filename in os.listdir(id_folder) if filename.startswith("result_TPEBO")]
        tpec_json_names[id] = [os.path.join(id_folder, filename) for filename in os.listdir(id_folder) if filename.startswith("result_TPEC")]

        # Natural sort for easier debugging 
        ea_json_names[id] = natsorted(ea_json_names[id])
        tpebo_json_names[id] = natsorted(tpebo_json_names[id])
        tpec_json_names[id] = natsorted(tpec_json_names[id])

        # NOTE: One task_id has 19 runs, not 20? Interesting - I think this was ID 10090
        # assert len(ea_json_names[id]) == non_random_replicates
        # assert len(tpebo_json_names[id]) == non_random_replicates
        # assert len(tpec_json_names[id]) == non_random_replicates

    ea_test_scores: dict[list] = {} # id: [scores...]
    tpebo_test_scores: dict[list] = {} 
    tpec_test_scores: dict[list] = {}

    ea_cv_scores: dict[list] = {}
    tpebo_cv_scores: dict[list] = {}
    tpec_cv_scores: dict[list] = {}

    # Compile EA scores
    for id, run_files in ea_json_names.items():  
        test_scores = []
        cv_scores = []
        for file in run_files:   
            with open(file, 'r') as f:
                data = json.load(f)
                test_scores.append(data["test_accuracy_score"])
                cv_scores.append(-data["cv_accuracy_score"])

        ea_test_scores[id] = test_scores
        ea_cv_scores[id] = cv_scores

    # Compile TPEBO scores
    for id, run_files in tpebo_json_names.items():  
        test_scores = []
        cv_scores = []
        for file in run_files:
            with open(file, 'r') as f:
                # test_score and cv_score are lists, each containing 2 copies of the same score
                test_score, cv_score = parse_multiple_json_for_score(file)
                test_scores.append(test_score[0]) # you only need one 
                cv_scores.append(cv_score[0])
        assert len(test_scores) == len(run_files)
        tpebo_test_scores[id] = test_scores
        tpebo_cv_scores[id] = cv_scores
    # print(tpebo_cv_scores)

    # Compile TPEC scores
    for id, run_files in tpec_json_names.items():  
        test_scores = []
        cv_scores = []
        for file in run_files:
            with open(file, 'r') as f:
                # test_score and cv_score are lists, each containing 2 copies of the same score
                test_score, cv_score = parse_multiple_json_for_score(file)
                test_scores.append(test_score[0]) # you only need one 
                cv_scores.append(cv_score[0])
        assert len(test_scores) == len(run_files)
        tpec_test_scores[id] = test_scores
        tpec_cv_scores[id] = cv_scores
    
    ea_records = []
    for task, test_scores in ea_test_scores.items():
        cv_scores = ea_cv_scores[task]
        for run, (test, cv) in enumerate(zip(test_scores, cv_scores)):
            ea_records.append({
                "task_id": task,
                "run": run,
                "test_score": test,
                "cv_score": cv
            })
    
    tpebo_records = []
    for task, test_scores in tpebo_test_scores.items():
        cv_scores = tpebo_cv_scores[task]
        for run, (test, cv) in enumerate(zip(test_scores, cv_scores)):
            tpebo_records.append({
                "task_id": task,
                "run": run,
                "test_score": test,
                "cv_score": cv
            })

    tpec_records = []
    for task, test_scores in tpec_test_scores.items():
        cv_scores = tpec_cv_scores[task]
        for run, (test, cv) in enumerate(zip(test_scores, cv_scores)):
            tpec_records.append({
                "task_id": task,
                "run": run,
                "test_score": test,
                "cv_score": cv
            })
    
    ea_df = pd.DataFrame(ea_records)
    tpebo_df = pd.DataFrame(tpebo_records)
    tpec_df = pd.DataFrame(tpec_records)
    return ea_df, tpebo_df, tpec_df

def merge(random_df, ea_df, tpebo_df, tpec_df):
    random_df["method"] = "Random"
    random_df = random_df[["task_id", "method", "run", "test_score", "cv_score"]]

    ea_df["method"] = "EA"
    ea_df = ea_df[["task_id", "method", "run", "test_score", "cv_score"]]

    tpebo_df["method"] = "TPEBO"
    tpebo_df = tpebo_df[["task_id", "method", "run", "test_score", "cv_score"]]

    tpec_df["method"] = "TPEC"
    tpec_df = tpec_df[["task_id", "method", "run", "test_score", "cv_score"]]

    merged_df = pd.concat([random_df, ea_df, tpebo_df, tpec_df], ignore_index=True)
    merged_df = merged_df.sort_values(["task_id", "method"])

    return merged_df

if __name__ == "__main__":
    # we did 15 reps for random, and 20 for the rest

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
    ea_df, tpebo_df, tpec_df = make_other_dfs(filtered_task_ids, other_results_path)

    random_df.to_csv("compiled_random.csv")
    ea_df.to_csv("compiled_ea.csv")
    tpebo_df.to_csv("compiled_tpebo.csv")
    tpec_df.to_csv("compiled_tpec.csv")

    merged_df = merge(random_df, ea_df, tpebo_df, tpec_df)
    merged_df.to_csv("all_method_scores_by_run.csv", index=False)