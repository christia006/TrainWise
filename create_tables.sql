CREATE TABLE IF NOT EXISTS experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    dataset_path VARCHAR(255) NOT NULL,
    target_column VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS models (
    model_id SERIAL PRIMARY KEY,
    experiment_id INT REFERENCES experiments(experiment_id),
    model_name VARCHAR(255) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    hyperparameters JSONB,
    training_time FLOAT,
    log_path VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(model_id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    split_type VARCHAR(50) NOT NULL
);