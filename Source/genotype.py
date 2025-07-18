from typeguard import typechecked
from abc import ABC, abstractmethod
import numpy as np

# create a base class that will hold our implementation of RandomForestClassifier
@typechecked
class ModelParams(ABC):
    def __init__(self, random_state: int, param_space: dict):
        self.random_state = random_state
        self.param_space = param_space
        return

    # function to get the model parameters
    def get_parameter_space(self):
        return self.param_space

    # function to generate a random set of parameters for the model
    @abstractmethod
    def generate_random_parameters(self):
        pass

    # function to get the specific type of a parameter within a model
    # must ignore the random_state parameter
    @abstractmethod
    def get_param_type(self, key: str):
        pass

    # function to shift float paramters either up or down
    def shift_float_parameter(self, cur_value: float, min: float, max: float, rng_: np.random.default_rng) -> float:
        rng = np.random.default_rng(rng_)
        # 68% of increases/decreases will be within 5% of the current value
        # 95% of increases/decreases will be within 10% of the current value
        # 99.7% of increases/decreases will be within 15% of the
        value = float(cur_value * rng.normal(1.0, 0.05))

        # ensure the value is within the bounds
        if value < min:
            return min
        elif value > max:
            return max
        else:
            return value

    # function to shift integer parameters either up or down
    def shift_int_parameter(self, cur_value: int, min: int, max: int, rng_: np.random.default_rng) -> int:
        rng = np.random.default_rng(rng_)
        # 68% of increases/decreases will be within 5% of the current value
        # 95% of increases/decreases will be within 10% of the current value
        # 99.7% of increases/decreases will be within 15% of the
        value = int(cur_value * rng.normal(1.0, 0.05))

        # ensure the value is within the bounds
        if value < min:
            return min
        elif value > max:
            return max
        else:
            return value

    # function to pick a new value from a categorical parameter
    def pick_categorical_parameter(self, choices: list, rng_: np.random.default_rng):
        rng = np.random.default_rng(rng_)
        # pick a new value from the choices
        return rng.choice(choices)

    # function to fix any parameters that do not align with scikit-learn's requirements
    @abstractmethod
    def fix_parameters(self, rng_: np.random.default_rng) -> None:
        pass

# create a RandomForest subclass that inherits from ModelParams
class RandomForestParams(ModelParams):
    def __init__(self, random_state: int, rng_: np.random.default_rng, params: dict = {}):
        rng = np.random.default_rng(rng_)
        self.model_params = {}
        self.param_space =  {
            'n_estimators': {'min': 10, 'max': 1000}, # int
            'criterion': ['gini', 'entropy', 'log_loss'], # categorical
            'max_depth': {'min': 1, 'max': 30}, # int
            'min_samples_split': {'min': .001, 'max': 1.0}, # float
            'min_samples_leaf': {'min': .001, 'max': 1.0}, # float
            'max_features': {'min': .001, 'max': 1.0}, # float
            'max_leaf_nodes': {'min': 2, 'max': 1000}, # int
            'bootstrap': [True, False],  # boolean
            'max_samples': {'min': .001, 'max': 1.0},  # float
            'random_state': random_state,  # int
        }
        super().__init__(random_state=random_state, param_space=self.param_space)

        # if params is empty, get a random set of parameters
        if len(params) == 0:
            self.model_params = self.generate_random_parameters(rng)
            # fix the parameters to ensure they are valid
            self.fix_parameters(rng)
        else:
            # otherwise, use the provided parameters
            self.model_params = params

    def mutate_parameters(self, rng_: np.random.default_rng) -> None:
        rng = np.random.default_rng(rng_)
        # update the models parameters by mutating them
        self.model_params = {
            'n_estimators': self.shift_int_parameter(self.model_params['n_estimators'], self.param_space['n_estimators']['min'], self.param_space['n_estimators']['max'], rng),
            'criterion': self.pick_categorical_parameter(self.param_space['criterion'], rng),
            'max_depth': self.shift_int_parameter(self.model_params['max_depth'], self.param_space['max_depth']['min'], self.param_space['max_depth']['max'], rng),
            'min_samples_split': self.shift_float_parameter(self.model_params['min_samples_split'], self.param_space['min_samples_split']['min'], self.param_space['min_samples_split']['max'], rng),
            'min_samples_leaf': self.shift_float_parameter(self.model_params['min_samples_leaf'], self.param_space['min_samples_leaf']['min'], self.param_space['min_samples_leaf']['max'], rng),
            'max_features': self.shift_float_parameter(self.model_params['max_features'], self.param_space['max_features']['min'], self.param_space['max_features']['max'], rng),
            'max_leaf_nodes': self.shift_int_parameter(self.model_params['max_leaf_nodes'], self.param_space['max_leaf_nodes']['min'], self.param_space['max_leaf_nodes']['max'], rng),
            'bootstrap': self.pick_categorical_parameter(self.param_space['bootstrap'], rng),
            'max_samples': self.shift_float_parameter(self.model_params['max_samples'], self.param_space['max_samples']['min'], self.param_space['max_samples']['max'], rng),
            'random_state': self.param_space['random_state'],  # keep the random state the same
        }
        # fix the parameters to ensure they are valid
        self.fix_parameters(rng)
        return

    def generate_random_parameters(self, rng_: np.random.default_rng) -> dict:
        rng = np.random.default_rng(rng_)
        return {
            'n_estimators': rng.integers(self.param_space['n_estimators']['min'], self.param_space['n_estimators']['max'], dtype=int),
            'criterion': rng.choice(self.param_space['criterion']),
            'max_depth': rng.integers(self.param_space['max_depth']['min'], self.param_space['max_depth']['max'], dtype=int),
            'min_samples_split': float(rng.uniform(self.param_space['min_samples_split']['min'], self.param_space['min_samples_split']['max'])),
            'min_samples_leaf': float(rng.uniform(self.param_space['min_samples_leaf']['min'], self.param_space['min_samples_leaf']['max'])),
            'max_features': float(rng.uniform(self.param_space['max_features']['min'], self.param_space['max_features']['max'])),
            'max_leaf_nodes': rng.integers(self.param_space['max_leaf_nodes']['min'], self.param_space['max_leaf_nodes']['max'], dtype=int),
            'bootstrap': rng.choice(self.param_space['bootstrap']),
            'max_samples': float(rng.uniform(self.param_space['max_samples']['min'], self.param_space['max_samples']['max'])),
            'random_state': self.param_space['random_state'],  # keep the random state the same
        }

    def get_param_type(self, key: str):
        if key in ['n_estimators', 'max_depth', 'max_leaf_nodes']:
            return 'int'
        elif key in ['min_samples_split', 'min_samples_leaf', 'max_features', 'max_samples']:
            return 'float'
        elif key in ['criterion', 'bootstrap']:
            return 'cat'
        else:
            print(f"Unknown parameter type for key: {key}")
            exit(-1)

    def fix_parameters(self, rng_: np.random.default_rng) -> None:
        # if bootstrap is False, we need to set max_samples to None
        if not self.model_params['bootstrap']:
            self.model_params['max_samples'] = None
        return