from src.load_data import load_data
from src.train import train_model
from src.mlflow_logging import setup_mlflow
from src.utils import load_config
from src.evaluate import evaluate_model
from src.model import get_model

import argparse
import mlflow
from transformers import pipeline


def main(config_path="configs/params.yaml", eval=False):
    config = load_config(config_path)
    setup_mlflow(config)

    model, tokenizer = get_model(config)
    with mlflow.start_run():
        # Загрузка данных
        train_dataset, val_dataset, test_dataset = load_data(tokenizer, config)
        print(f"Loaded dataset, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
        # Обучение
        if not eval:
            model = train_model(model, train_dataset, val_dataset, config)
        # Оценка на тесте
        evaluate_model(model, test_dataset, config)
        # Создание пайплайна
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        # Логирование лучшей модели
        mlflow.transformers.log_model(pipe, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", "-e", help="evaluate only", action="store_true"
    )
    parser.add_argument(
        "--config",
        "-c",
        help="set config file",
        type=str,
        default="configs/params.yaml",
    )
    args = parser.parse_args()
    main(args.config, args.eval)
