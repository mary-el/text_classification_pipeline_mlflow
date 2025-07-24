import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    fbeta_score
)
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
FBETA_LABELS = [0, 1, 3, 4, 5, 6, 7]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "f1": f1_score(labels, predictions, average="weighted"),
        "fbeta": fbeta_score(labels, predictions, average="weighted", beta=0.5, labels=FBETA_LABELS)
    }


def evaluate_model(model, test_dataset, config):
    # Предсказания модели
    loader = DataLoader(test_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    model.eval()

    y_test = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_test.extend(labels.cpu().numpy())
    # Вычисление метрик
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, average="weighted"),
        "test_recall": recall_score(y_test, y_pred, average="weighted"),
        "test_f1": f1_score(y_test, y_pred, average="weighted"),
        "test_fbeta": fbeta_score(y_test, y_pred, average="weighted", beta=0.5, labels=FBETA_LABELS)
    }

    # Логирование метрик в MLflow
    mlflow.log_metrics(metrics)
    # Логирование результатов теста
    pd.DataFrame({'true_label': y_test, 'pred_label': y_pred}).to_csv("results.csv")
    mlflow.log_artifact('results.csv', artifact_path="reports")
    # Генерация и сохранение confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, config['model']['labels'])  

    # Логирование classification report
    report = classification_report(y_test, y_pred, output_dict=True, target_names=config['model']['labels'])
    pd.DataFrame(report).transpose().to_csv("classification_report.csv")
    mlflow.log_artifact("classification_report.csv", artifact_path="reports")


def save_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Сохранение в файл
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Логирование в MLflow
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
