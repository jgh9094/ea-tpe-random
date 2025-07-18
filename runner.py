import Source.experiment as experiment
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True,)
    parser.add_argument("--n_jobs", type=int, required=True,)
    parser.add_argument("--save_path", type=str, required=True,)
    parser.add_argument("--seed", type=int, required=True,)
    parser.add_argument("--data_dir", type=str, required=True,)
    args = parser.parse_args()

    # execute the experiment with the provided arguments
    experiment.execute_experiment(args.task_id, args.n_jobs, args.save_path, args.seed, args.data_dir)