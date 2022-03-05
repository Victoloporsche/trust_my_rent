import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np


class FeatureEngineering:
    def __init__(self, input_data, target_col):
        self.input_data = input_data
        self.target_col = target_col
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def get_categorical_features(self) -> pd.DataFrame:
        categorical_features = [feature for feature in self.input_data.columns if
                                self.input_data[feature].dtypes == "O"]
        categorical_features_data = self.input_data[categorical_features]
        return categorical_features_data

    def _input_rare_categorical(self) -> pd.DataFrame:
        categorical_features = self.get_categorical_features()
        for feature in categorical_features:
            temp = self.input_data.groupby(feature)[self.target_col].count() / len(
                self.input_data
            )
            temp_df = temp[temp > 0.01].index
            self.input_data[feature] = np.where(
                self.input_data[feature].isin(temp_df),
                self.input_data[feature],
                "Rare_var",
            )
            return self.input_data

    def _label_encode_cat_features(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        categorical_features = self.get_categorical_features()

        mapping_dict = {}
        for feature in categorical_features:
            processed_df[feature] = self.label_encoder.fit_transform(processed_df[feature])
            cat_mapping = dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            )
            mapping_dict[feature] = cat_mapping

        with open("data/processed/label_encoding.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in mapping_dict.items():
                writer.writerow([key, value])
        return processed_df

    def _fill_na_missing_categorical(self) -> pd.DataFrame:
        df = self._input_rare_categorical()
        nan_categorical = [
            feature
            for feature in df.columns
            if df[feature].isnull().sum() > 0 and df[feature].dtypes == "O"
        ]
        df[nan_categorical] = df[nan_categorical].fillna("Missing")
        return df

    def _fill_na_missing_numerical(self) -> pd.DataFrame:
        df = self._fill_na_missing_categorical()
        numerical_with_nan = [
            feature
            for feature in df.columns
            if df[feature].isnull().sum() > 0 and df[feature].dtypes != "O"
        ]
        df[numerical_with_nan] = df[numerical_with_nan].fillna(
            df[numerical_with_nan].mean()
        )

        return df

    def _scale_features(self) -> pd.DataFrame:
        processed_df = self._fill_na_missing_numerical()
        df = self._label_encode_cat_features(processed_df)
        scaling_feature = [
            feature for feature in df.columns if feature not in [self.target_col]
        ]
        scaling_features_data = df[scaling_feature]
        self.scaler.fit(scaling_features_data)
        self.scaler.transform(scaling_features_data)

        return pd.concat(
            [
                df[[self.target_col]].reset_index(drop=True),
                pd.DataFrame(
                    self.scaler.transform(df[scaling_feature]), columns=scaling_feature
                ),
            ],
            axis=1,
        )

