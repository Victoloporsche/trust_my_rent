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
    test_size = config['base']['test_size']
    random_state = config['base']['random_state']
    k_value = config['base']['k_value']

    optim = ModelOptimization(input_data=input_data,
                              target_col=target_col)
    best_parameters = optim.perform_grid_search_model_optimization(test_size=test_size,
                                                                   random_state=random_state,
                                                                   k_value=k_value)
    return best_parameters


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    optimal_parameters = get_parameters(config_path=parsed_args.config)
    print(optimal_parameters)
