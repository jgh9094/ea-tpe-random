"""
Script to compile average results from the Random, EA, TPEBO, TPEC RF experiments.
Produces "all_scores.csv".
Requires "filtered_random_scores.csv".
"""

import pandas as pd
import os
import re
import numpy as np
import json

# Path structure (non-random experiments): './rfresults/{TASK ID}/result_{METHOD}_{RUN}.jsonl
# all_scores.csv structure:
# method | task_id | avg_test_score | std_test_score

# function to parse a single file of multiple JSON objects for the test score
def parse_multiple_json_for_score(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split at '}{' but keep the braces using a positive lookahead/lookbehind
    json_objects = re.split(r'(?<=\})\s*(?=\{)', content)

    test_scores = []
    for i, obj in enumerate(json_objects):
        try:
            # Fix "False"/"True" stringified booleans if needed
            obj_fixed = re.sub(r'"\s*(True|False)\s*"', lambda m: m.group(1).lower(), obj)

            data = json.loads(obj_fixed)
            test_scores.append(data["test_accuracy_score"])
        except Exception as e:
            print(f"Error parsing object #{i+1}: {e}")
            continue

    return test_scores

def dict_to_df(score_dict, method_name):
    df = pd.DataFrame.from_dict(score_dict, orient='index', columns=["avg_test_score", "std_test_score"])
    df.index.name = "task_id"
    df = df.reset_index()
    df["method"] = method_name
    return df[["method", "task_id", "avg_test_score", "std_test_score"]] # change order of columns

# all_scores is long format
def merge(random_df, ea_scores, tpebo_scores, tpec_scores, output_path):
    # long_df = pd.DataFrame(columns=["method", "task_id", "avg_test_score", "std_test_score"])
    # methods = ["Random", "EA", "TPEBO", "TPEC"]
    random_df["method"] = "Random"
    random_df = random_df[["method", "task_id", "avg_test_score", "std_test_score"]]
    
    ea_df = dict_to_df(ea_scores, "EA")
    tpebo_df = dict_to_df(tpebo_scores, "TPEBO")
    tpec_df = dict_to_df(tpec_scores, "TPEC")

    merged_df = pd.concat([random_df, ea_df, tpebo_df, tpec_df], ignore_index=True)
    merged_df = merged_df.sort_values(["task_id", "method"])

    merged_df.to_csv(output_path, index=False)
    




non_random_replicates = 20 # we did 15 reps for random, and 20 for the rest

# removed task IDs with less than 15 runs
random_scores_path = 'results/filtered_random_scores.csv' 
random_scores_df = pd.read_csv(random_scores_path)

# this directory stores results from EA, TPEBO, and TPEC
other_results_path = 'rfresults/'

# Retrieve names of hardest tasks (less than 80% accuracy on test set)
random_scores_hard_df = random_scores_df[random_scores_df['avg_test_score'] < 0.8]
task_ids = random_scores_hard_df['task_id'].tolist()
task_ids.sort()

# Collect paths to JSON files for each method 
ea_json_names: dict[list] = {} # id: [filenames...]
tpebo_json_names: dict[list] = {}
tpec_json_names: dict[list] = {}

# Iterate through folders in 'rfresults'
for id in task_ids:
    id_folder = os.path.join(other_results_path, str(id))
    if not os.path.isdir(id_folder): continue # if a specific id does not exist in 'rfresults', skip

    ea_json_names[id] = [os.path.join(id_folder, filename) for filename in os.listdir(id_folder) if filename.startswith("result_EA")]
    tpebo_json_names[id] = [os.path.join(id_folder, filename) for filename in os.listdir(id_folder) if filename.startswith("result_TPEBO")]
    tpec_json_names[id] = [os.path.join(id_folder, filename) for filename in os.listdir(id_folder) if filename.startswith("result_TPEC")]

    # One task_id has 19 runs, not 20? Interesting - I think this was ID 10090
    # assert len(ea_json_names[id]) == non_random_replicates
    # assert len(tpebo_json_names[id]) == non_random_replicates
    # assert len(tpec_json_names[id]) == non_random_replicates

ea_stats: dict[tuple] = {} # id: (avg_test_score, std_test_score)
tpebo_stats: dict[tuple] = {} 
tpec_stats: dict[tuple] = {} 

# Compile EA stats
for id, run_files in ea_json_names.items():  
    test_scores = []
    for file in run_files:   
        with open(file, 'r') as f:
            data = json.load(f)
            test_scores.append(data["test_accuracy_score"])
    
    assert len(test_scores) == len(run_files)
    ea_stats[id] = (np.mean(test_scores), np.std(test_scores))
#print(ea_stats)

# Compile TPEBO stats
for id, run_files in tpebo_json_names.items():  
    test_scores = []
    for file in run_files:
        with open(file, 'r') as f:
            test_score, _ = parse_multiple_json_for_score(file)
            test_scores.append(test_score)
    assert len(test_scores) == len(run_files)
    tpebo_stats[id] = (np.mean(test_scores), np.std(test_scores))
# print(tpebo_stats)

# # Compile TPEC stats
for id, run_files in tpec_json_names.items():  
    test_scores = []
    for file in run_files:
        with open(file, 'r') as f:
            test_score, _ = parse_multiple_json_for_score(file)
            test_scores.append(test_score)
    assert len(test_scores) == len(run_files)
    tpec_stats[id] = (np.mean(test_scores), np.std(test_scores))
# print(tpec_stats)

merge(random_scores_hard_df, ea_stats, tpebo_stats, tpec_stats, "results/all_scores.csv")
# NOTE: theres a task id that only exists in random 