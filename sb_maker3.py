# 10/24/25 - redo random runs
# clear; python sb_maker.py > runner.sb
# must make first elif -> if

import pandas as pd

if __name__ == "__main__":    # Example usage
    # read csv file
    df = pd.read_csv('data/task_list.csv')
    df = df.sort_values(by='rows', ascending=True)

    # get task_id column as list
    task_ids = df['task_id'].tolist()

    lower_bound = 1
    upper_bound = 10
    offset = 10

    lower_bound_list = []
    upper_bound_list = []
    task_id_list = []

    # header prints
    print("#!/bin/bash")
    print("########## Define Resources Needed with SBATCH Lines ##########")
    print("#SBATCH --nodes=1")
    print("#SBATCH --ntasks=1")
    print("#SBATCH --array=1-710")
    print("#SBATCH --cpus-per-task=48")
    print("#SBATCH -t 02:00:00")
    print("#SBATCH --mem=100GB")
    print("#SBATCH --job-name=rng_rf")
    print("#SBATCH -p defq")
    print("###############################################################\n")

    print("source ~/anaconda3/etc/profile.d/conda.sh")
    print("conda init")
    print("conda activate tpe-ea\n")

    print('DATA_DIR=/mnt/home/suzuekar/ea-tpe-random/data/')
    print("RESULTS_DIR=/mnt/home/suzuekar/ea-tpe-random/results/")
    print("mkdir -p ${RESULTS_DIR}\n")

    print('##################################')
    print('# Treatments')
    print('##################################\n')

    for task_id in task_ids:
        lower_bound_list.append(f'TASK_{task_id}_MIN')
        print(f'TASK_{task_id}_MIN={lower_bound}')
        upper_bound_list.append(f'TASK_{task_id}_MAX')
        print(f'TASK_{task_id}_MAX={upper_bound}')

        lower_bound += offset
        upper_bound += offset
        task_id_list.append(task_id)

    print('##################################')
    print('# Conditions')
    print('##################################\n')

    for lower_bound_str, upper_bound_str, task_id in zip(lower_bound_list, upper_bound_list, task_id_list):
        print(f'elif [ ${{SLURM_ARRAY_TASK_ID}} -ge ${lower_bound_str} ] && [ ${{SLURM_ARRAY_TASK_ID}} -le ${upper_bound_str} ] ; then')
        print(f'    TASK_ID={task_id}')
        print(f'    REP_DIR=${{RESULTS_DIR}}/${{TASK_ID}}/Rep_${{SLURM_ARRAY_TASK_ID}}/')
    print('else')
    print('  echo "${SEED} from ${TASK_ID} failed to launch"')
    print('fi')
    print()
    print("mkdir -p ${REP_DIR}\n")
    print("# todo: let it rip")
    print("python runner.py \\")
    print("-task_id ${TASK_ID} \\")
    print("-n_jobs 48 \\")
    print("-save_path ${REP_DIR} \\")
    print("-seed ${SLURM_ARRAY_TASK_ID} \\")
    print("-data_dir ${DATA_DIR} \\")
