# 工具文件
'''
accuracy
f1
seed
模型保存
SwanLab 可视化
'''
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def compute_f1(preds, labels, average="macro"):
    return f1_score(labels, preds, average=average)


def compute_precision(preds, labels, average="macro"):
    return precision_score(labels, preds, average=average)


def compute_recall(preds, labels, average="macro"):
    return recall_score(labels, preds, average=average)


def count_labels(path):
    from collections import Counter
    counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label = int(line.split("\t")[0])
            counter[label] += 1
    return counter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
