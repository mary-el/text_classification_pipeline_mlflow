import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(config):
    id2label = {i: key for i, key in enumerate(config["model"]["labels"])}
    label2id = {val: key for key, val in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["pretrained_name"],
        num_labels=len(id2label), id2label=id2label, label2id=label2id
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_name"])
    return model, tokenizer
