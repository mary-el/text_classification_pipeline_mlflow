import mlflow
import numpy as np
import pandas as pd

from src.dataset import TextClassificationDataset


def process_texts(texts, tokenizer, config):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=config["data"]["max_text_length"]
    )
    return encodings


def load_data(tokenizer, config):
    df = pd.read_csv(config["data"]["path"], encoding='utf-8', index_col=0)

    mlflow.log_artifact(config["data"]["path"], artifact_path='data')

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]

    if "split_column" in config["data"]:  # если сплит не задан, задаём сами
        split_column = config["data"]["split_column"]
    else:
        print("Creating split")
        split_column = "split"
        df[split_column] = np.random.choice(['train', 'val', 'test'], len(df), p=[0.7, 0.2, 0.1])

    df[text_column] = df[text_column].fillna('').astype(str)

    train_ids = df[split_column] == 'train'
    val_ids = df[split_column] == 'val'
    test_ids = df[split_column] == 'test'

    df_train = df[train_ids]
    df_val = df[val_ids]
    df_test = df[test_ids]

    texts_train = df_train[text_column].tolist()
    texts_val = df_val[text_column].tolist()
    texts_test = df_test[text_column].tolist()

    labels_train = df_train[label_column].tolist()
    labels_val = df_val[label_column].tolist()
    labels_test = df_test[label_column].tolist()

    encodings_train = process_texts(texts_train, tokenizer, config)
    encodings_val = process_texts(texts_val, tokenizer, config)
    encodings_test = process_texts(texts_test, tokenizer, config)

    classes = config["model"]["labels"]
    train_dataset = TextClassificationDataset(encodings_train, labels_train, classes)
    val_dataset = TextClassificationDataset(encodings_val, labels_val, classes)
    test_dataset = TextClassificationDataset(encodings_test, labels_test, classes)

    return train_dataset, val_dataset, test_dataset
