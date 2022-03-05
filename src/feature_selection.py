from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

from feature_engineering import FeatureEngineering
import pandas as pd


class FeatureSelection:
    def __init__(self, input_data, target_col):
        self.input_data = input_data
        self.target_col = target_col
        self.feature_engineering = FeatureEngineering(input_data, target_col)
        self.output = self.input_data[self.target_col]

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scaled_data = self.feature_engineering._scale_features()
        train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

        return scaled_data, train_data, test_data

    def _perform_feature_selection(self, num_of_features_to_select: int = None) -> pd.DataFrame:
        scaled_data, train, _ = self._split_data()

        if not num_of_features_to_select:
            num_of_features_to_select = len(scaled_data.columns)

        y_train = train[self.target_col]
        x_train = train.drop([self.target_col], axis=1)

        feature_sel_model = ExtraTreesRegressor().fit(x_train, y_train)
        feat_importance = pd.Series(
            feature_sel_model.feature_importances_, index=x_train.columns
        )
        selected_features = feat_importance.nlargest(num_of_features_to_select)
        selected_features_df = selected_features.to_frame()
        selected_features_list = selected_features_df.index.tolist()
        features_df = scaled_data[selected_features_list]
        return pd.concat([self.output, features_df], axis=1)
