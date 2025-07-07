# automl_platform/db_manager.py

import psycopg2
from psycopg2 import extras
import json
import os
from datetime import datetime

class DBManager:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Connected to PostgreSQL database!")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def disconnect(self):
        if self.conn:
            self.conn.close()
            print("Disconnected from PostgreSQL database.")

    def _execute_query(self, query, params=None, fetch_results=False):
        if not self.conn:
            self.connect()
            if not self.conn:
                return None

        try:
            with self.conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch_results:
                    return cur.fetchall()
                else:
                    self.conn.commit()
                    return cur.rowcount
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            self.conn.rollback()
            return None

    def insert_experiment(self, experiment_name, dataset_path, target_column, notes=None):
        query = """
        INSERT INTO experiments (experiment_name, dataset_path, target_column, notes)
        VALUES (%s, %s, %s, %s) RETURNING experiment_id;
        """
        result = self._execute_query(query, (experiment_name, dataset_path, target_column, notes), fetch_results=True)
        return result[0]['experiment_id'] if result else None

    def update_experiment_status(self, experiment_id, status, end_time=None):
        query = """
        UPDATE experiments SET status = %s, end_time = %s WHERE experiment_id = %s;
        """
        self._execute_query(query, (status, end_time, experiment_id))

    def insert_model(self, experiment_id, model_name, model_path, hyperparameters, training_time, log_path):
        query = """
        INSERT INTO models (experiment_id, model_name, model_path, hyperparameters, training_time, log_path)
        VALUES (%s, %s, %s, %s::jsonb, %s, %s) RETURNING model_id;
        """
        result = self._execute_query(query, (experiment_id, model_name, model_path, json.dumps(hyperparameters), training_time, log_path), fetch_results=True)
        return result[0]['model_id'] if result else None

    def insert_metric(self, model_id, metric_name, metric_value, split_type):
        query = """
        INSERT INTO metrics (model_id, metric_name, metric_value, split_type)
        VALUES (%s, %s, %s, %s);
        """
        self._execute_query(query, (model_id, metric_name, metric_value, split_type))

    def get_experiments(self):
        query = "SELECT * FROM experiments ORDER BY start_time DESC;"
        return self._execute_query(query, fetch_results=True)

    def get_models_for_experiment(self, experiment_id):
        query = "SELECT * FROM models WHERE experiment_id = %s;"
        return self._execute_query(query, (experiment_id,), fetch_results=True)

    def get_metrics_for_model(self, model_id):
        query = "SELECT * FROM metrics WHERE model_id = %s;"
        return self._execute_query(query, (model_id,), fetch_results=True)