import openml
import sklearn
import numpy as np
import os
import pickle
import pandas
import tpot

#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, preprocess=True, classification=True):

    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        num_classes = len(set(y))

        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)

            # needed this to LabelEncode the target variable if it is a classification task only
            if classification:
                le = sklearn.preprocessing.LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, task.openml_url, num_classes, y_test

def main():
    # save the dimensions of the data
    task_ids = []
    features = []
    rows = []
    urls = []
    classes = []

    # Classification tasks from the 'AutoML Benchmark All Classification' suite
    # Suite is used within 'AMLB: an AutoML Benchmark' paper
    # https://github.com/openml/automlbenchmark
    # https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf
    # https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=271
    for i, task_id in enumerate(openml.study.get_suite('271').tasks):
        print(f'{i}: {task_id}')
        X_train, url, c, _= load_task(task_id)

        # save task id, number of features, and number of instances
        task_ids.append(task_id)
        features.append(X_train.shape[1])
        rows.append(X_train.shape[0])
        urls.append(url)
        classes.append(c)
    print('Finished classification tasks')

    # save the data to a csv file
    df = pandas.DataFrame({'task_id': task_ids, 'features': features, 'rows': rows, 'classes': classes, 'url': urls})
    df.to_csv('task_list.csv', index=False)

if __name__ == '__main__':
    main()
    print('FINISHED')