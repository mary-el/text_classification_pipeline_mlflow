import os

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
from scipy.special import softmax
from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(eval_pred, beta = 1, other_class = None):
    logits, labels = eval_pred
    FBETA_LABELS = list(range(logits.shape[1]))

    if not other_class is None:
        FBETA_LABELS.remove(other_class)
        
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "fbeta": fbeta_score(labels, predictions, average="weighted", beta=beta, labels=FBETA_LABELS)
    }


def visualize_pr(metrics, classes, best_thresholds, beta, results_path):
    n_classes = len(classes)
    # Визуализация для каждого класса
    fig, axes = plt.subplots(n_classes, 3, figsize=(18, 4 * n_classes))
    thresholds = np.arange(0.05, 1.0, 0.01)


    for i, cls in enumerate(classes):
        # Precision
        axes[i, 0].plot(thresholds, metrics['precision'][i], label='Precision', color='blue')
        axes[i, 0].axvline(best_thresholds[i], color='red', linestyle='--', 
                        label=f'Best threshold: {best_thresholds[i]:.2f}')
        axes[i, 0].set_title(f'Class {cls} Precision')
        axes[i, 0].legend()
        
        # Recall
        axes[i, 1].plot(thresholds, metrics['recall'][i], label='Recall', color='green')
        axes[i, 1].axvline(best_thresholds[i], color='red', linestyle='--')
        axes[i, 1].set_title(f'Class {cls} Recall')
        axes[i, 1].legend()
        
        # Fβ-score
        axes[i, 2].plot(thresholds, metrics['fbeta'][i], label=f'F{beta}-score', color='purple')
        axes[i, 2].axvline(best_thresholds[i], color='red', linestyle='--')
        axes[i, 2].set_title(f'Class {cls} F{beta}-score')
        axes[i, 2].legend()

    plt.tight_layout()
    plt.savefig(f"{results_path}/PR.png")
    plt.close()
    mlflow.log_artifact(f"{results_path}/PR.png", artifact_path="plots")


def find_best_thresholds(df, class_other, n_classes, beta=1):
    # Диапазон порогов
    thresholds = np.arange(0.05, 1.0, 0.01)
    classes = list(range(n_classes))

    metrics = {
        'precision': {cls: [] for cls in classes},
        'recall': {cls: [] for cls in classes},
        'fbeta': {cls: [] for cls in classes}
    }

    best_thresholds = {cls: 0 for cls in classes}
    best_fbeta_scores = {cls: 0 for cls in classes}
    true_labels = df['true_label'].to_list()

    # Вычисляем метрики для каждого порога и каждого класса
    for cls in classes:
        for threshold in thresholds:
            # Временные предсказания: если вероятность < порога, то other (2), иначе исходный класс
            y_pred_temp = np.where(df['probs'] < threshold, class_other, df['pred_label'])
            
            # Для текущего класса вычисляем precision и recall
            precision = precision_score(true_labels, y_pred_temp, labels=[cls], average='micro', zero_division=0)
            recall = recall_score(true_labels, y_pred_temp, labels=[cls], average='micro', zero_division=0)
            fbeta = fbeta_score(true_labels, y_pred_temp, beta=beta, labels=[cls], average='micro', zero_division=0)
            
            # Сохраняем метрики
            metrics['precision'][cls].append(precision)
            metrics['recall'][cls].append(recall)
            metrics['fbeta'][cls].append(fbeta)
            # Обновляем лучший порог для текущего класса
            if fbeta > best_fbeta_scores[cls]:
                best_fbeta_scores[cls] = fbeta
                best_thresholds[cls] = threshold
    # Вывод оптимальных порогов

    print(f"Optimal thresholds for each class (using F{beta}-score):")
    for cls, threshold in best_thresholds.items():
        print(f"Class {cls}: {threshold:.2f} (F{beta}-score: {best_fbeta_scores[cls]:.3f})")

    return best_thresholds, metrics



def evaluate_model(model, test_dataset, config):
    # Предсказания модели
    loader = DataLoader(test_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    model.eval()

    y_test = []
    logits = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits

            logits.extend(outputs.cpu().numpy())
            y_test.extend(labels.cpu().numpy())

    logits = np.stack(logits, axis=0)
    y_pred = np.argmax(logits, axis=-1)  #   предсказанные метки
    probs = softmax(logits, axis=-1).max(axis=-1)   # вероятности для предсказанных меток

    classes = config["model"]["labels"]
    other_class = config["data"]["other_class"]
    beta = config["training"].get("beta", 1)
    results_path = config["evaluation"]["results_path"]
    os.makedirs(results_path, exist_ok=True)

    # Вычисление метрик
    metrics = compute_metrics((logits, y_test), other_class=other_class, beta=beta)
    
    # Логирование метрик в MLflow
    mlflow.log_metrics(metrics)
    # Логирование результатов теста
    df = pd.DataFrame({'true_label': y_test, 'pred_label': y_pred, 'probs': probs})
    # Лучшие пороги для F-beta
    best_thresholds, metrics_by_thresholds = find_best_thresholds(df, other_class, len(classes), beta)
    visualize_pr(metrics_by_thresholds, classes, best_thresholds, beta, results_path)

    # Применяем пороги к тестовым данным
    df['final_pred'] = df['pred_label'].copy()
    
    thresholds = config["evaluation"].get("thresholds", [0,] * len(classes))
    print(f"Your thresholds: {thresholds}")
    
    for cls, threshold in enumerate(thresholds):
        if cls != other_class:  # Не применяем порог к other классу (2)
            df.loc[(df['pred_label'] == cls) & (df['probs'] < threshold), 'final_pred'] = other_class

    df.to_csv(f"{results_path}/results.csv")
    mlflow.log_artifact(f"{results_path}/results.csv", artifact_path="reports")
    # Генерация и сохранение confusion matrix
    cm = confusion_matrix(y_test, df["final_pred"])
    save_confusion_matrix(cm, classes, results_path=results_path, thresholds=thresholds)  

    # Логирование classification report
    report = classification_report(y_test, df["final_pred"], output_dict=True, target_names=classes)
    pd.DataFrame(report).transpose().to_csv(f"{results_path}/classification_report.csv")
    mlflow.log_artifact(f"{results_path}/classification_report.csv", artifact_path="reports")


def save_confusion_matrix(cm, class_names, results_path, thresholds):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel(f"Predicted Label; Thresholds={thresholds}")

    # Сохранение в файл
    plt.savefig(f"{results_path}/confusion_matrix.png")
    plt.close()

    # Логирование в MLflow
    mlflow.log_artifact(f"{results_path}/confusion_matrix.png", artifact_path="plots")
    mlflow.log_dict({"confusion_matrix": cm.tolist()}, f"{results_path}/confusion_matrix.json")
