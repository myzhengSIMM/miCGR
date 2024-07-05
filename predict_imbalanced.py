import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'

from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from core.model import NSWSSC_k2, NSWSSC_k3, NSWSSC_k4, NSWSSC_k5, NSWSSC_k6, NSWSSC_k7
from core.dataset import TrainDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from utils.sequence import load_train, make_test_input_pair, postprocess_result, make_test_input_pair2
from utils.normalizers import ChannelNormalizer
from utils.early_stop import EarlyStopping
import os
import pickle
import sys




cuda = "cuda:3"
save_odir = "results_imbalanced"
kmer=6
cts_size=30
weight_file = "./model/model_weights.pt"
level = "gene"
mirna_fasta_ifile = "./imbalanced_data/hsa_mirna.fa"
mrna_fasta_ifile = "./imbalanced_data/longest_utr.fa"
if not os.path.exists(save_odir):
    os.makedirs(save_odir)

def calculation_scores(outputs, labels):
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    roc_auc = metrics.roc_auc_score(labels, outputs)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, outputs)
    pr_auc = metrics.auc(recall, precision)
    return roc_auc, pr_auc

def evaluate_sc(tn, fp, fn, tp):
    try:
        sen = round(tp / (tp + fn), 4)
    except:
        sen = 0
    try:
        spe = round(tn / (tn + fp), 4)
    except:
        spe = 0
    try:
        ppv = round(tp / (tp + fp), 4)
    except:
        ppv = 0
    try:
        npv = round(tn / (tn + fn), 4)
    except:
        npv = 0
    return sen, spe, ppv, npv

record_fout = open(os.path.join(save_odir, "records_outtest2.csv"), 'w')
record_fout.write(','.join(['split', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'PPV', 'NPV'])+'\n')

for seed_match in ["offset-9-mer-m7"]:
    print("[{:s}]".format(seed_match))
    record_fout.write("{:s}\n".format(seed_match))
    query_file = "unbalanced/test.txt"
    output_file = save_odir + "/imbalanced_predicts_{:s}.csv".format(seed_match)
    if kmer == 2:
        model = NSWSSC_k2(dropout=dropout)
    elif kmer == 3:
        model = NSWSSC_k3(dropout=dropout)
    elif kmer == 4:
        model = NSWSSC_k4(dropout=dropout)
    elif kmer == 5:
        model = NSWSSC_k5(dropout=dropout)
    elif kmer == 6:
        model = NSWSSC_k6(dropout=dropout)
    elif kmer == 7:
        model = NSWSSC_k7(dropout=dropout)
    else:
        raise ValueError("'kmer' expected '2-7' integer, got '{}'".format(kmer))
    model.load_state_dict(torch.load(weight_file, map_location='cuda:0'))
    dataset, neg_dataset = make_test_input_pair(mirna_fasta_ifile,
                                                mrna_fasta_ifile,
                                                query_file,
                                                 cts_size=cts_size,
                                                 kmer=kmer,
                                                 seed_match=seed_match,
                                                 header=True,
                                                 train=True,
                                                 fix=False,
                                                 cutoff=1)
    mirna_arr = dataset['query_arr']
    mrna_arr = dataset['target_arr']
    mirna = torch.FloatTensor(mirna_arr)
    mrna = torch.FloatTensor(mrna_arr)
    device = torch.device(cuda)
    model.to(device)
    with torch.no_grad():
        model.eval()
        outputs = model.inference([mirna, mrna], batch_size, device)
    _, predicts = torch.max(outputs.data, 1)
    probabilities = F.softmax(outputs, dim=1)

    y_probs = probabilities.cpu().numpy()[:, 1]
    y_predicts = predicts.cpu().numpy()

    results = postprocess_result(dataset, neg_dataset, y_probs, y_predicts,
                                 cts_size=cts_size, level=level, output_file=output_file)
    tn, fp, fn, tp = metrics.confusion_matrix(results['LABEL'], results['PREDICT']).ravel()
    acc = (tp + tn) / len(results['LABEL'])
    sensitivity = tp / (tp + fn)  # true positive rate
    specificity = tn / (tn + fp)  # true negative rate
    f1 = metrics.f1_score(results['LABEL'], results['PREDICT'])
    roc_auc = metrics.roc_auc_score(results['LABEL'], results['PROBABILITY'])
    precision, recall, thresholds = metrics.precision_recall_curve(results['LABEL'], results['PROBABILITY'])
    pr_auc = metrics.auc(recall, precision)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print(
        "ACC: {:.4f}, ROC-AUC: {:.4f}, PR-AUC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, F1: {:.4f}, PPV: {:.4f}, NPV: {:.4f}".format(
            acc, roc_auc, pr_auc, sensitivity, specificity, f1, ppv, npv))
    record_fout.write("test{:d},{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
        1, acc, sensitivity, specificity, f1, ppv, npv))
        # break
    record_fout.write('\n')
    record_fout.write('\n')
    # break
record_fout.close()
