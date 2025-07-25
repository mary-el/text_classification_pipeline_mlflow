import time

from functools import partial

import mlflow
from transformers import Trainer, TrainingArguments
from transformers.integrations import MLflowCallback

from src.text_classification.evaluate import compute_metrics


def train_model(model, train_data, val_data, config):
    training_args = TrainingArguments(
        output_dir=config["training"]["output_path"] + '/' + time.strftime("%Y-%m-%d-%H-%M-%S"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(config["training"]["batch_size"]),
        num_train_epochs=int(config["training"]["num_epochs"]),
        weight_decay=float(config["training"]["weight_decay"]),
        save_total_limit=int(config["training"]["save_total_limit"]),
        logging_dir=config["training"]["logs"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        report_to="mlflow",
        load_best_model_at_end=True,
    )

    compute_metrics_partial = partial(compute_metrics, other_class=config["training"].get("other_class", None),
                              beta=config["training"].get("beta", 1))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics_partial,
        callbacks=[MLflowCallback()]
    )
    mlflow.log_params({**config["training"], **config["data"], **config["model"]})
    trainer.train()

    return model
