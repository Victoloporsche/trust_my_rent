from model_optimization import ModelOptimization
import pandas as pd
import yaml
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path) -> pd.DataFrame:
    config = read_params(config_path)
    data_path = config['data_source']['sql_source']
    df = pd.read_csv(data_path, sep=',', encoding='utf-8')
    return df


def get_parameters(config_path) -> dict:
    config = read_params(config_path)
    sql_data = config['data_source']['sql_source']
    input_data = pd.read_csv(sql_data)

    target_col = config['base']['target_col']

    optim = ModelOptimization(input_data=input_data,
                              target_col=target_col)
    best_parameters = optim.perform_grid_search_model_optimization()
    return best_parameters


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    optimal_parameters = get_parameters(config_path=parsed_args.config)
    print(optimal_parameters)
