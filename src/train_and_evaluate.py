from sklearn.linear_model import Ridge
import pandas as pd
import yaml
import argparse
import joblib
import json
from urllib.parse import urlparse
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)
    data_path = config['data_source']['sql_source']
    df = pd.read_csv(data_path, sep=',', encoding='utf-8')
    return df


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_path = config['train_test_path']['train_data']
    test_path = config['train_test_path']['test_data']
    target_col = config['base']['target_col']
    random_state = config['base']['random_state']
    alpha = config["estimators"]["RidgeRegression"]["params"]["alpha"]

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    y_train = train_data[target_col]
    x_train = train_data.drop([target_col], axis=1)
    y_test = test_data[target_col]
    x_test = test_data.drop([target_col], axis=1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        lr = Ridge(
            alpha=alpha,
            random_state=random_state)
        lr.fit(x_train, y_train)

        predicted_qualities = lr.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        mlflow.log_param("alpha", alpha)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr, "model")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
