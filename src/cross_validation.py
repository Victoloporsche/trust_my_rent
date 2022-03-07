from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score

from feature_selection import FeatureSelection
import pandas as pd


class CrossValidation:
    def __init__(self, input_data, target_col, num_of_features_to_select=None):
        self.num_of_features_to_select = num_of_features_to_select
        self.input_data = input_data
        self.target_col = target_col

        self.feature_selection = FeatureSelection(input_data, target_col)
        self.clf_models = list()
        self._initialize_clf_models()

    def _train_test_split(self, test_size, random_state) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features_df = self.feature_selection._perform_feature_selection(test_size,
                                                                        random_state,
                                                                        self.num_of_features_to_select)
        train_data, test_data = train_test_split(features_df, test_size=test_size, random_state=random_state)
        train_data.to_csv('data/processed/train.csv', index=False)
        test_data.to_csv('data/processed/test.csv', index=False)

        y_train = train_data[self.target_col]
        x_train = train_data.drop([self.target_col], axis=1)
        y_test = test_data[self.target_col]
        x_test = test_data.drop([self.target_col], axis=1)
        return x_train, y_train, x_test, y_test

    def _get_models(self) -> list:
        return self.clf_models

    def _add(self, model) -> None:
        self.clf_models.append((model))

    def _initialize_clf_models(self) -> None:
        model = RandomForestRegressor()
        self.clf_models.append((model))

        model = ExtraTreesRegressor()
        self.clf_models.append((model))

        model = ElasticNet()
        self.clf_models.append((model))

        model = linear_model.Lasso()
        self.clf_models.append((model))

        model = linear_model.Ridge()
        self.clf_models.append((model))

        base_learners = [("lr", LinearRegression()), ("ET", ExtraTreesRegressor())]
        model = StackingRegressor(
            estimators=base_learners, final_estimator=linear_model.Ridge()
        )
        self.clf_models.append((model))

        model = AdaBoostRegressor()
        self.clf_models.append((model))

        model = GradientBoostingRegressor()
        self.clf_models.append((model))

    def _k_fold_cross_validation(self, test_size, random_state, k_value: int):
        x_train, y_train, x_test, y_test = self._train_test_split(test_size, random_state)
        clf_models = self._get_models()
        models = []
        results = {}

        for model in clf_models:
            current_model_name = model.__class__.__name__

            cross_validate = cross_val_score(model, x_train, y_train, cv=k_value)
            mean_cross_validation_score = cross_validate.mean()
            print(
                f"K_fold cross validation for {current_model_name} is {mean_cross_validation_score}"
            )
            results[current_model_name] = mean_cross_validation_score
            models.append(model)

        best_model = max(results, key=lambda k: results.get(k))

        print(f"The best model for this dataset is {best_model}")

        return best_model, x_train, y_train, x_test, y_test
