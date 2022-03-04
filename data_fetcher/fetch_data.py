import mysql.connector
from mysql.connector import Error
import yaml
import argparse
from sqlalchemy import create_engine
import pandas as pd


class SqlDatabase:
    def __init__(self, config_path):
        self.config_path = config_path
        self.connection = self.connection()

    def _read_params(self):
        with open(self.config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config

    def connection(self):
        config = self._read_params()
        return mysql.connector.connect(host=config['database']['host'],
                                       database=config['database']['detabase_name'],
                                       user=config['database']['user'],
                                       password=config['database']['password'])

    def connect_to_database(self):
        try:
            if self.connection.is_connected():
                db_Info = self.connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)
                cursor = self.connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print("You're connected to database: ", record)
        except Error as e:
            print("Error while connecting to MySQL", e)

    def read_table_in_database(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(" ".join(["select * from", self._read_params()['database']['table_name']]))
            records = cursor.fetchall()
            print(records)
            print("Total number of rows in table: ", cursor.rowcount)

        except mysql.connector.Error as e:
            print("Error reading data from MySQL table", e)

    def database_table_to_df(self):
        db_connection_str = "".join(["mysql+pymysql://", self._read_params()['database']['user'],
                                     ":", self._read_params()['database']['password'],
                                     "@", self._read_params()['database']['host'],
                                     "/", self._read_params()['database']['detabase_name']])
        db_connection = create_engine(db_connection_str)

        df = pd.read_sql(" ".join(["select * from", self._read_params()['database']['table_name']]), con=db_connection)
        return df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='config_file.yaml')
    parsed_args = args.parse_args()
    sql_database = SqlDatabase(config_path=parsed_args.config)
    df = sql_database.database_table_to_df()
    print(df)
