from models.model_trc import *
import random
import csv
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch


def calculate_average_metrics(results):
    metrics = list(results[0].keys()) if results else []
    avg_results = {metric: 0.0 for metric in metrics}

    for result in results:
        for metric in metrics:
            avg_results[metric] += result[metric]

    num_results = len(results)
    if num_results > 0:
        for metric in metrics:
            avg_results[metric] /= num_results

    return avg_results


def metric_aupr(labels, preds):
    return average_precision_score(labels.numpy(), preds.numpy())


def train(model, device, train_loader, optimizer, epoch):
    lossz = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y.view(-1, 1).float())

        loss.backward()
        optimizer.step()
        lossz += loss.item()

    print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, lossz))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    return total_labels, total_preds


def metric(labels, preds):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
        matthews_corrcoef

    labels_np = labels.numpy()
    preds_np = preds.numpy()

    sig = 1.0 / (1.0 + np.exp(-preds_np))
    binary_preds = (sig >= 0.5).astype(int)

    try:
        auc_ = roc_auc_score(labels_np, sig)
    except ValueError:
        auc_ = 0.0

    acc_ = accuracy_score(labels_np, binary_preds)
    pre_ = precision_score(labels_np, binary_preds, zero_division=0)
    rec_ = recall_score(labels_np, binary_preds, zero_division=0)

    try:
        f1_ = f1_score(labels_np, binary_preds, zero_division=0)
    except ValueError:
        f1_ = 0.0

    mcc_ = matthews_corrcoef(labels_np, binary_preds) if len(np.unique(binary_preds)) > 1 else 0.0

    return auc_, acc_, pre_, rec_, f1_, mcc_


def set_seed(seed_value=42):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    cuda_name = "cuda:0"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    NUM_EPOCHS = 500
    LR = 0.001

    print('cuda_name:', cuda_name)
    print('Learning rate:', LR)
    print('Epochs:', NUM_EPOCHS)

    root = 'data_rlm'
    processed_train = root + '/train.pth'
    processed_test = root + '/test.pth'
    data_listtrain = torch.load(processed_train)
    data_listtest = torch.load(processed_test)


    def custom_batching(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]


    def batch_data(data_list):
        return Batch.from_data_list(data_list)


    batchestrain = list(custom_batching(data_listtrain, 256))
    batchestrain1 = []
    for batch in batchestrain:
        batchestrain1.append(batch_data(batch))

    batchestest = list(custom_batching(data_listtest, 1000))
    batchestest1 = []
    for batch in batchestest:
        batchestest1.append(batch_data(batch))

    model = MSTRC(
        num_features_xd=43,
        hidden_dim=128,
        heads=4,
        dropout=0.2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    all_metrics = []
    max_auc = 0

    for epoch in range(NUM_EPOCHS):
        train(model, device, batchestrain1, optimizer, epoch + 1)
        G, P = predicting(model, device, batchestest1)

        auc_, acc_, precision_, recall_, f1_, mcc_ = metric(G, P)
        aupr_ = metric_aupr(G, P)

        epoch_result = {
            'AUC': auc_,
            'AUPR': aupr_,
            'ACC': acc_,
            'FI': f1_,
            'REC': recall_,
            'PRE': precision_,
            'MCC': mcc_
        }
        all_metrics.append(epoch_result)

        if auc_ > max_auc:
            max_auc = auc_
            ret1 = [auc_, aupr_, acc_, precision_, recall_, f1_, mcc_]

        print('Epoch:%3d | AUC:%.4f AUPR:%.4f ACC:%.4f PRE:%.4f REC:%.4f F1:%.4f MCC:%.4f' %
              (epoch + 1, auc_, aupr_, acc_, precision_, recall_, f1_, mcc_))

    avg_metrics = calculate_average_metrics(all_metrics)

    all_metrics_sorted = sorted(all_metrics, key=lambda x: x['AUC'], reverse=True)
    top_k = 5 if len(all_metrics_sorted) >= 5 else len(all_metrics_sorted)
    top5_metrics = all_metrics_sorted[:top_k]
    avg_top5 = calculate_average_metrics(top5_metrics)

    print(f"\n[Top-{top_k} Epochs by AUC] Average Metrics")
    for metric_key, value in avg_top5.items():
        print(f"{metric_key}: {value:.4f}")

    print("\nTraining completed, calculating final test set metrics...")

    final_labels, final_preds = predicting(model, device, batchestest1)
    final_auc, final_acc, final_precision, final_recall, final_f1, final_mcc = metric(final_labels, final_preds)
    final_aupr = metric_aupr(final_labels, final_preds)

    print("\nFinal test set evaluation results:")
    print("AUC: {:.4f} | AUPR: {:.4f} | ACC: {:.4f} | PRE: {:.4f} | REC: {:.4f} | F1: {:.4f} | MCC: {:.4f}".format(
        final_auc, final_aupr, final_acc, final_precision, final_recall, final_f1, final_mcc
    ))