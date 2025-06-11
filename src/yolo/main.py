import argparse
import os
import re

import mlflow
from dotenv import load_dotenv
from ultralytics import YOLO, settings

from src.mlflow_logging import setup_mlflow
from src.utils import load_config


def main(config_path: str, eval: bool = False):
    load_dotenv()
    settings.update({"mlflow": True})

    config = load_config(config_path)
    setup_mlflow(config)

    os.environ["MLFLOW_EXPERIMENT_NAME"] = config["logging"]["experiment_name"]
    model_name = config['model']['name']
    model_register_name = config['model']['register_name']
    data_path = os.path.join(config['data']['location'], 'data.yaml')
    train_params = config['training']
    pretrained = config['model']['pretrained']

    with mlflow.start_run():
        run = mlflow.active_run()
        os.environ["MLFLOW_RUN"] = run.info.run_name

        # Загрузка модели
        if pretrained:
            model = YOLO(f'{model_name}.pt')  # Загрузка предобученной модели
        else:
            model = YOLO(f'{model_name}.yaml')  # Создание новой модели из конфига
        if not eval:
            # Обучение модели
            results = model.train(
                data=data_path,
                verbose=True,
                **train_params
            )

            # Логирование метрик
            for metric_name, metric_values in results.results_dict.items():
                metric_name = re.sub(metric_name, r'[^a-zA-Z0-9_-./ ]', ' ')
                if isinstance(metric_values, (int, float)):
                    mlflow.log_metric(metric_name, metric_values)
                elif isinstance(metric_values, dict):
                    for k, v in metric_values.items():
                        mlflow.log_metric(f"{metric_name}_{k}", v)

            # Логирование артефактов
            mlflow.log_artifact(os.path.join(model.trainer.save_dir, 'weights/best.pt'), 'model')
            mlflow.log_artifact(config_path, 'config')
            # Логирование модели в MLflow Model Registry
            mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path=model_register_name,
                registered_model_name=model_register_name,
                # input_example=np.ndarray((3, imgsz, imgsz), dtype=float)
            )
            print(f"Обучение завершено. Результаты сохранены в MLflow: {mlflow.get_artifact_uri()}")
        metrics = model.val(
            data=data_path,
            split="test",
            plots=True,  # сохранить графики (PR-кривые, confusion matrix)
            save_json=True,  # сохранить метрики в JSON
            save_hybrid=True  # гибридный режим (предсказания + GT)
        )

        # Логируем метрики
        mlflow.log_metrics({
            "test_mAP50": metrics.box.map50,
            "test_mAP50-95": metrics.box.map,
            "test_precision": metrics.box.p.mean(),
            "test_recall": metrics.box.r.mean(),
            "test_f1": metrics.box.f1.mean()
        })


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
