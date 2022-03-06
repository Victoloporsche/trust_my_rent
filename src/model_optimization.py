from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from cross_validation import CrossValidation
from sklearn.model_selection import GridSearchCV
import numpy as np

np.random.seed(42)


class ModelOptimization:
    def __init__(self, input_data, target_col, num_of_features_to_select=None):
        self.num_of_features_to_select = num_of_features_to_select
        self.input_data = input_data
        self.target_col = target_col
        self.cross_validation = CrossValidation(input_data, target_col, num_of_features_to_select)

    def perform_grid_search_model_optimization(self):
        best_model, x_train, y_train, x_test, y_test = self.cross_validation._k_fold_cross_validation()

        if best_model == "ExtraTreesRegressor":
            ex = ExtraTreesRegressor()
            criterion = ["mse", "mae"]
            num_estimators = [50, 100]

            parameters = {"n_estimators": num_estimators, "criterion": criterion}

            optim = GridSearchCV(ex, param_grid=parameters)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters

        elif best_model == "RandomForestRegressor":
            rf = RandomForestRegressor()
            criterion = ["mse", "mae"]
            num_estimators = [50, 100]

            parameters = {"n_estimators": num_estimators, "criterion": criterion}

            optim = GridSearchCV(rf, param_grid=parameters)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters

        elif best_model == "ElasticNet":
            elastic = ElasticNet()
            params = {"alpha": [0.02, 0.024, 0.025, 0.026, 0.03]}

            optim = GridSearchCV(elastic, param_grid=params)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters


        elif best_model == "Lasso":
            lasso = linear_model.Lasso()
            params = {"alpha": [0.02, 0.024, 0.025, 0.026, 0.03]}

            optim = GridSearchCV(lasso, param_grid=params)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters

        elif best_model == "Ridge":
            ridge = linear_model.Ridge()
            params = {"alpha": [0.02, 0.024, 0.025, 0.026, 0.03]}

            optim = GridSearchCV(ridge, param_grid=params)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters

        elif best_model == "StackingRegressor":
            base_learners = [("lr", LinearRegression()), ("ET", ExtraTreesRegressor())]

            stacking = StackingRegressor(base_learners)

            fit_model = stacking.fit(x_train, y_train)


        elif best_model == "AdaBoostRegressor":
            adaboost = AdaBoostRegressor()

            params = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
                "loss": ["linear", "square", "exponential"],
            }

            optim = GridSearchCV(adaboost, param_grid=params)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters

        elif best_model == "GradientBoostingRegressor":
            gb = GradientBoostingRegressor()
            params = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
                "loss": ["ls", "lad", "huber", "quantile"],
            }

            optim = GridSearchCV(gb, param_grid=params)
            optim.fit(x_train, y_train)
            optimal_parameters = optim.best_params_.items()
            return optimal_parameters
        else:
            raise ValueError('The best model does not have an optimization strategy!')
