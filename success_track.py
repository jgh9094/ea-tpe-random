"""
Directory structure:
    'results/{task_id}/Rep_{slurm_array_id}/{task_id}-{slurm_id}/results.csv'

Each 'results.csv' has: 
    task_id,seed,best_cv_score,test_score

Output CSV ("completion_summary.csv"): 
    task_id, seed, best_cv_score, test_score, finished
"""
import pandas as pd
import os

if __name__ == "__main__":
    results_path = 'results/'
    task_list_path = 'data/task_list.csv'

    df = pd.read_csv(task_list_path)
    df = df.sort_values(by='rows', ascending=True)
    task_ids = df['task_id'].tolist()

    compiled_rows = [] # list of dicts

    for task_id in task_ids:
        # results/{task_id}
        task_dir = os.path.join(results_path, str(task_id))
        print(task_dir)
        if not os.path.exists(task_dir):
            print(f"Folder for task {task_id} doesn't exist.")
            continue

        # results/{task_id}/Rep_{slurm_id}
        for rep_folder in os.listdir(task_dir):
            # ignore folders that don't start with "Rep_"
            if not rep_folder.startswith("Rep_"): continue
            rep_path = os.path.join(task_dir, rep_folder)
            print(rep_path)
            if not os.path.isdir(rep_path): 
                print(f"Folder {rep_path} doesn't exist.")
                continue
            
            # {task_id}/Rep_{slurm_id}/{task_id}-{slurm_id}/results.csv
            for sub in os.listdir(rep_path):
                print(sub)
                  
                sub_path = os.path.join(rep_path, sub)
                print(sub_path)
                results_csv_path = os.path.join(sub_path, "results.csv")
                
                print(results_csv_path)
                if os.path.exists(results_csv_path): 
                    try: # there should only be a single row
                        results_row = pd.read_csv(results_csv_path).iloc[0]
                        compiled_rows.append({
                            "task_id": task_id,
                            "seed": results_row.get("seed"),
                            "best_cv_score": results_row.get("best_cv_score"),
                            "test_score": results_row.get("test_score"),
                            "finished": 1
                        })
                    except Exception as e:
                        print(f"Error reading {results_csv_path} - {e}")
                
                else:
                    compiled_rows.append({
                        "task_id": task_id,
                        "seed": rep_folder.replace("Rep_", ""),
                        "best_cv_score": None,
                        "test_score": None,
                        "finished": 0
                    })

    summary_df = pd.DataFrame(compiled_rows)
    # summary_df = summary_df.sort_values(by=["task_id", "replicate"])
    print("Columns:", summary_df.columns.tolist())
    print("Number of rows:", len(summary_df))
    print(summary_df.head())
    summary_df.to_csv("completion_summary.csv", index=False)




