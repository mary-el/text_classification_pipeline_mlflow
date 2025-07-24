import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnx

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

def save_model(model, tokenizer, config):
    model.save_pretrained(config["model"]["save_to"])
    tokenizer.save_pretrained(config["model"]["save_to"])
    print(f"Saved model to {config["model"]["save_to"]}")

def save_onnx(config, tokenizer):
    ort_model = ORTModelForSequenceClassification.from_pretrained(config["model"]["save_to"], export=True)
    ort_model.save_pretrained(config["model"]["onnx_path"])
    tokenizer.save_pretrained(config["model"]["onnx_path"])

    onnx_checkpoint = f'{config["model"]["onnx_path"]}/model.onnx'
    model = onnx.load(onnx_checkpoint)
    for input_tensor in model.graph.input:
        input_tensor.type.tensor_type.elem_type = onnx.TensorProto.INT32
    onnx.save(model, onnx_checkpoint)
