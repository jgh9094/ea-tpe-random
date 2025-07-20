# clear; python sb_maker.py > runner.sb
# must make first elif -> if

import pandas as pd

if __name__ == "__main__":    # Example usage
    # read csv file
    df = pd.read_csv('/common/suzuek/ea-tpe-random/data/task_list.csv')

    # get task_id column as list
    task_ids = df['task_id'].tolist()

    lower_bound = 1
    upper_bound = 15
    offset = 15

    lower_bound_list = []
    upper_bound_list = []
    task_id_list = []

    # header prints
    print("#!/bin/bash")
    print("########## Define Resources Needed with SBATCH Lines ##########")
    print("#SBATCH --nodes=1")
    print("#SBATCH --ntasks=1")
    print("#SBATCH --array=1-1065")
    print("#SBATCH --cpus-per-task=8")
    print("#SBATCH -t 24:00:00")
    print("#SBATCH --mem=100GB")
    print("#SBATCH --job-name=rng_rf")
    print("#SBATCH -p defq")
    print("#SBATCH --exclude=esplhpc-cp040")
    print("###############################################################\n")

    print("# todo: load conda environment")
    print("source /common/suzuek/miniconda3/etc/profile.d/conda.sh")
    print("conda activate tpe-ea\n")

    print("# todo: define the output and data directory")
    print('DATA_DIR=/common/suzuek/ea-tpe-random/data/')
    print("RESULTS_DIR=/common/suzuek/ea-tpe-random/results/")
    print("mkdir -p ${RESULTS_DIR}\n")

    print('##################################')
    print('# Treatments')
    print('##################################\n')

    for task_id in task_ids:
        lower_bound_list.append(f'TASK_{task_id}_MIN')
        print(f'TASK_{task_id}_MIN={lower_bound}')
        upper_bound_list.append(f'TASK_{task_id}_MAX')
        print(f'TASK_{task_id}_MAX={upper_bound}')
        print()

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
    print("python /common/suzuek/ea-tpe-random/runner.py \\")
    print("-task_id ${TASK_ID} \\")
    print("-n_jobs 8 \\")
    print("-save_path ${REP_DIR} \\")
    print("-seed ${SLURM_ARRAY_TASK_ID} \\")
    print("-data_dir ${DATA_DIR} \\")
