import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for genotype, env, labels, cond in dataloader:
            genotype, env, labels = genotype.to(device), env.to(device), labels.to(device)
            logits, _, _ = model(genotype, env)
            probs = torch.softmax(logits, dim=1)[:,1]

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    return acc, f1, auroc

def print_metrics(acc, f1, auroc, prefix=""):
    print(f"{prefix} - Acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
