import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
import ray
from Source.genotype import RandomForestParams
from sklearn.model_selection import StratifiedKFold
import pandas as pd

CV_K = 5
REPLICATES = 20

# evaluate unseen snps
@ray.remote
def ray_eval_random_forest(x_train,
                           y_train,
                           cv,
                           n_estimators: int,
                           criterion: str,
                           max_depth: int,
                           min_samples_split: float,
                           min_samples_leaf: float,
                           max_features: float,
                           max_leaf_nodes: int,
                           bootstrap: bool,
                           max_samples: float,
                           random_state: int,
                           i: int) -> float:
    accuracies = []
    for train_index, test_index in cv.split(x_train, y_train):
        # create a random forest model with the given parameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=random_state
        )

        # split the data into training and testing sets
        x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # fit the model on the training set
        try:
            model.fit(x_train_fold, y_train_fold)
        except Exception as e:
            print(f"Error fitting model {i}: {e}")
            return -1.0, i

        # evaluate the model on the test set
        score = model.score(x_test_fold, y_test_fold)
        accuracies.append(score)

    # return the mean accuracy and the model
    return np.mean(accuracies, dtype=float), i

#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id: int, data_dir: str, preprocess=True):
    cached_data_path = f"{data_dir}/{task_id}_{preprocess}.pkl"
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        print(f'Task {task_id} not found')
        exit(0)

    return X_train, y_train, X_test, y_test

# execute task with tpot2
def execute_experiment(task_id: int, n_jobs: int, save_path: str, seed: int, data_dir: str):
    # variables
    total_evals = 10000

    # initialize ray
    ray.init(num_cpus=n_jobs, include_dashboard=True)

    # initialize the random number generator
    rng = np.random.default_rng(seed)

    # generate directory to save results
    save_folder = f"{save_path}/{seed}-{task_id}"
    if os.path.exists(save_folder):
        print('FOLDER ALREADY EXISTS:', save_folder)
        return

    print("LOADING DATA")
    X_train, y_train, X_test, y_test = load_task(task_id, data_dir, preprocess=True)
    X_train_id = ray.put(X_train)
    y_train_id = ray.put(y_train)
    print(f"DATA LOADED: {X_train.shape[0]} TRAINING SAMPLES, {X_test.shape[0]} TEST SAMPLES")

    # initialize our cv splitter
    cv = StratifiedKFold(n_splits=CV_K, shuffle=True, random_state=seed)

    # create total_evals number of random forest models
    print("CREATING MODELS")
    all_models = [RandomForestParams(random_state=seed, rng_=rng, params={}) for _ in range(total_evals)]
    scores = [-1.0] * total_evals
    print(f"CREATED {len(all_models)} MODELS")

    # evaluate each model in parallel
    print("EVALUATING MODELS")
    ray_jobs = []
    for i, model in enumerate(all_models):
        ray_jobs.append(ray_eval_random_forest.remote(
            x_train=X_train_id,
            y_train=y_train_id,
            cv=cv,
            n_estimators= model.model_params['n_estimators'],
            criterion=model.model_params['criterion'],
            max_depth=model.model_params['max_depth'],
            min_samples_split=model.model_params['min_samples_split'],
            min_samples_leaf=model.model_params['min_samples_leaf'],
            max_features=model.model_params['max_features'],
            max_leaf_nodes=model.model_params['max_leaf_nodes'],
            bootstrap=model.model_params['bootstrap'],
            max_samples=model.model_params['max_samples'],
            random_state=model.model_params['random_state'],
            i=i
        ))
    assert len(ray_jobs) == total_evals, "Number of jobs does not match total_evals"

    # process results as they come in
    while len(ray_jobs) > 0:
        finished, ray_jobs = ray.wait(ray_jobs)
        r2, i = ray.get(finished)[0]
        scores[i] = r2

    # count number of failed models (score == -1.0)
    print(f"NUMBER OF FAILED MODELS: {scores.count(-1.0)}")

    # get the indexs of the models with the best scores
    best_score = max(scores)
    best_models = [i for i, score in enumerate(scores) if score == best_score]
    print(f"BEST SCORE: {best_score} FOR {len(best_models)} MODELS")

    # randomly pick an index from the best models
    best_model_index = rng.choice(best_models)
    assert scores[best_model_index] == best_score, "Best model score does not match best score"

    # train the best model on the full training set and evaluate on the test set
    best_model_params = all_models[best_model_index]

    model = RandomForestClassifier(**best_model_params.model_params)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # create pandas dataframe to save results
    results = pd.DataFrame({
        'task_id': [task_id],
        'seed': [seed],
        'best_cv_score': [best_score],
        'test_score': [test_score]
    })
    # save results to csv
    results.to_csv(f"{save_folder}/results.csv", index=False)

    return