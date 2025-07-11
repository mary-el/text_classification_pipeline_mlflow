import mlflow


def setup_mlflow(config):
    mlflow.set_tracking_uri(config["logging"]["uri"])
    mlflow.set_experiment(config["logging"]["experiment_name"])
    mlflow.transformers.autolog()
