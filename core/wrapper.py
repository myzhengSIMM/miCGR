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
from utils.sequence import load_train, make_test_input_pair, postprocess_result, make_test_input_pair2, make_test_input_pair3
from utils.normalizers import ChannelNormalizer
from utils.early_stop import EarlyStopping
import os

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


def train_model_crossval(train_file,
                kmer=4,
                learning_rate=0.001,
                batch_size=32,
                epochs=50,
                save_odir=None,
                device='cuda',
                nfold=10,
                test_size=0.2,
                dropout=0.2,
                patience=100,
                seed=123):
    print("\n[TRAIN] {:s}-L{}-B{:d}-E{:d}_S{:d}_D{}_TS{}_crossval".format(
        "NSWSSC", learning_rate, batch_size, epochs, seed, dropout, test_size))
    train_dat_dict = load_train(train_file)
    mirna_arr, mrna_arr, label_arr = train_dat_dict["mirna"], train_dat_dict["mrna"], train_dat_dict["label"]
    x_train_idx, x_test_idx, left_label_arr, test_label_arr = train_test_split(np.arange(len(label_arr)), label_arr,
                                                                               stratify=label_arr,
                                                                               random_state=seed,
                                                                               test_size=test_size)
    left_mirna_arr, left_mrna_arr = mirna_arr[x_train_idx], mrna_arr[x_train_idx]
    test_mirna_arr, test_mrna_arr = mirna_arr[x_test_idx], mrna_arr[x_test_idx]
    skf = StratifiedKFold(n_splits=nfold, random_state=seed, shuffle=True)
    device = torch.device(device)
    record_file = os.path.join(save_odir, "record_L{}_B{:d}_E{:d}_S{:d}_D{}_TS{}.csv".format(
        learning_rate, batch_size, epochs, seed, dropout, test_size))
    log_file = os.path.join(save_odir, "record_L{}_B{:d}_E{:d}_S{:d}_D{}_TS{}.log".format(
        learning_rate, batch_size, epochs, seed, dropout, test_size))
    record_fout = open(record_file, 'w')
    log_fout = open(log_file, 'w')
    record_fout.write(','.join(
        ['split', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'PPV', 'NPV', 'best_epochs', 'best_val_acc']) + '\n')
    for n, (train_idx, val_idx) in enumerate(skf.split(left_label_arr, left_label_arr)):
        start_time = datetime.now()
        print("[Fold {:d} start] {}".format(n, start_time.strftime('%Y-%m-%d @ %H:%M:%S')))
        log_fout.write("[Fold {:d} start] {}\n".format(n, start_time.strftime('%Y-%m-%d @ %H:%M:%S')))
        train_mirna_arr, train_mrna_arr, train_label_arr = left_mirna_arr[train_idx], left_mrna_arr[train_idx], \
                                                           left_label_arr[train_idx]
        val_mirna_arr, val_mrna_arr, val_label_arr = left_mirna_arr[val_idx], left_mrna_arr[val_idx], left_label_arr[
            val_idx]
        weight_file = os.path.join(save_odir, "weight_fold{:d}_L{}_B{:d}_E{:d}_S{:d}_D{}_TS{}.pt".format(
            n, learning_rate, batch_size, epochs, seed, dropout, test_size))
        train_set = TrainDataset(train_mirna_arr, train_mrna_arr, train_label_arr)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_mirna = torch.FloatTensor(val_mirna_arr)
        val_mrna = torch.FloatTensor(val_mrna_arr)
        val_label = torch.LongTensor(val_label_arr).to(device)
        train_mirna = torch.FloatTensor(train_mirna_arr)
        train_mrna = torch.FloatTensor(train_mrna_arr)
        train_label = torch.LongTensor(train_label_arr).to(device)
        test_mirna = torch.FloatTensor(test_mirna_arr)
        test_mrna = torch.FloatTensor(test_mrna_arr)
        test_label = torch.LongTensor(test_label_arr).to(device)
        class_weight = torch.Tensor(compute_class_weight(class_weight='balanced',
                                                         classes=np.unique(train_label_arr),
                                                         y= train_label_arr)).to(device)
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
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
        early_stopping= EarlyStopping(patience=patience,
                                      delta=0.001,
                                      param_name="Val_ACC",
                                      mode='max',
                                      verbose=False,
                                      path=weight_file)
        for epoch in range(epochs):
            model.train()
            with tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, epochs), bar_format=bar_format) as tqdm_loader:
                for i, ((mirna, mrna), label) in enumerate(tqdm_loader):
                    mirna, mrna, label = mirna.to(device, dtype=torch.float), mrna.to(device, dtype=torch.float), label.to(device)
                    batch_pred = model((mirna, mrna))
                    loss = criterion(batch_pred, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (i+1) == len(train_loader):
                        model.eval()
                        with torch.no_grad():
                            train_outputs = model.inference([train_mirna, train_mrna],
                                                            128,
                                                            device)
                            val_outputs = model.inference([val_mirna, val_mrna],
                                                          128,
                                                          device)
                            test_outputs = model.inference([test_mirna, test_mrna],
                                                           128,
                                                           device)
                        train_loss = criterion(train_outputs, train_label).item()
                        val_loss = criterion(val_outputs, val_label).item()
                        test_loss = criterion(test_outputs, test_label).item()
                        # train_loss_lst.append(train_loss)
                        # val_loss_lst.append(val_loss)
                        # test_loss_lst.append(test_loss)
                        _, train_pred = torch.max(train_outputs, 1)
                        _, val_pred = torch.max(val_outputs, 1)
                        _, test_pred = torch.max(test_outputs, 1)
                        train_prob = F.softmax(train_outputs, dim=1)[:, 1]
                        val_prob = F.softmax(val_outputs, dim=1)[:, 1]
                        test_prob = F.softmax(test_outputs, dim=1)[:, 1]
                        train_roc_auc, train_pr_auc = calculation_scores(train_prob, train_label)
                        val_roc_auc, val_pr_auc = calculation_scores(val_prob, val_label)
                        test_roc_auc, test_pr_auc = calculation_scores(test_prob, test_label)
                        train_f1 = metrics.f1_score(train_label.cpu().detach().numpy(),
                                                    train_pred.cpu().detach().numpy())
                        val_f1 = metrics.f1_score(val_label.cpu().detach().numpy(), val_pred.cpu().detach().numpy())
                        test_f1 = metrics.f1_score(test_label.cpu().detach().numpy(), test_pred.cpu().detach().numpy())
                        train_corrects = (train_pred == train_label).sum().item()
                        val_corrects = (val_pred == val_label).sum().item()
                        test_corrects = (test_pred == test_label).sum().item()
                        train_acc = train_corrects / len(train_pred)
                        val_acc = val_corrects / len(val_pred)
                        test_acc = test_corrects / len(test_pred)
                        val_tn, val_fp, val_fn, val_tp = metrics.confusion_matrix(val_label.cpu().detach().numpy(),
                                                                                  val_pred.cpu().detach().numpy()).ravel()
                        test_tn, test_fp, test_fn, test_tp = metrics.confusion_matrix(test_label.cpu().detach().numpy(),
                                                                                      test_pred.cpu().detach().numpy()).ravel()
                        val_sen, val_spe, val_ppv, val_npv = evaluate_sc(val_tn, val_fp, val_fn, val_tp)
                        test_sen, test_spe, test_ppv, test_npv = evaluate_sc(test_tn, test_fp, test_fn, test_tp)
                        tqdm_loader.set_postfix(
                            dict(train_loss=train_loss,
                                 val_loss=val_loss,
                                 test_loss=test_loss,
                                 val_acc=val_acc))
                        early_stopping(val_acc, model)
                        if early_stopping.saving_flag:
                            best_val_acc = val_acc
                            best_epoch = epoch + 1
                            best_test = [test_acc, test_sen, test_spe, test_f1, test_ppv, test_npv]
                            if save_odir is not None:
                                torch.save(model.state_dict(), weight_file)
                        log_fout.write(
                            "train_loss={:.6f}, val_loss={:.6f}, test_loss={:.6f}, val_acc={:.4f}\n".format(
                                train_loss, val_loss, test_loss, val_acc))
                    else:
                        tqdm_loader.set_postfix(loss=loss.item())

                # print("[Epoch {}]".format(epoch + 1))
                #     # print("train_roc_auc={}, train_pr_auc={}, train_acc={}".format(train_roc_auc, train_pr_auc, train_acc))
                # print(
                #     "val_roc_auc={:.4f}, val_pr_auc={:.4f}, val_acc={:.4f}, sensitivity={:.4f}, specificity={:.4f}, val_f1={:.4f}, PPV={:.4f}, NPV={:.4f}".format(
                #         val_roc_auc, val_pr_auc, val_acc, val_sen, val_spe, val_f1, val_ppv, val_npv))
                # print(
                #     "test_roc_auc={:.4f}, test_pr_auc={:.4f}, test_acc={:.4f}, sensitivity={:.4f}, specificity={:.4f},test_f1={:.4f}, PPV={:.4f}, NPV={:.4f}".format(
                #         test_roc_auc, test_pr_auc, test_acc, test_sen, test_spe, test_f1, test_ppv, test_npv))
                print(
                    "val_roc_auc={:.4f}, val_pr_auc={:.4f}, val_acc={:.4f}, sensitivity={:.4f}, specificity={:.4f}, val_f1={:.4f}, PPV={:.4f}, NPV={:.4f}".format(
                        val_roc_auc, val_pr_auc, val_acc, val_sen, val_spe, val_f1, val_ppv, val_npv), file=log_fout)
                print(
                    "test_roc_auc={:.4f}, test_pr_auc={:.4f}, test_acc={:.4f}, sensitivity={:.4f}, specificity={:.4f},test_f1={:.4f}, PPV={:.4f}, NPV={:.4f}".format(
                        test_roc_auc, test_pr_auc, test_acc, test_sen, test_spe, test_f1, test_ppv, test_npv),
                    file=log_fout)
            if early_stopping.stopping_flag:
                break
        print("Sample{:d} Achieve best performance at epoch {:d} with best Acc {}.".format(n, best_epoch,
                                                                                           best_val_acc))
        print("Sample{:d} Achieve best performance at epoch {:d} with best Acc {}.".format(n, best_epoch,
                                                                                           best_val_acc), file=log_fout)
        record_fout.write(
            'F{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:d},{:.4f}\n'.format(
                n, best_test[0], best_test[1], best_test[2], best_test[3], best_test[4], best_test[5], best_epoch,
                best_val_acc))
        end_time = datetime.now()
        print("\n[Fold {:d} finish] {} (used time: {})\n".format(n, end_time.strftime('%Y-%m-%d @ %H:%M:%S'),
                                                       (end_time - start_time)))
        # break
    record_fout.close()
    log_fout.close()


def train_model_integrate(train_file,
                kmer=4,
                learning_rate=0.001,
                batch_size=32,
                epochs=50,
                save_odir=None,
                dropout=0.2,
                device='cuda'):
    print("\n[TRAIN] {:s}-L{}-B{:d}-E{:d}_integrate".format("NotShareWeightedSiameseSC",
                                                  learning_rate,
                                                  batch_size,
                                                  epochs))
    log_file = os.path.join(save_odir, "record_L{}_B{:d}_E{:d}_S{:d}.log".format(learning_rate,
                                                                                 batch_size,
                                                                                 epochs,
                                                                                 seed))
    log_fout = open(log_file, 'w')
    train_dat_dict = load_train(train_file)
    mirna_arr, mrna_arr, label_arr = train_dat_dict["mirna"], train_dat_dict["mrna"], train_dat_dict["label"]
    device = torch.device(device)
    weight_file = os.path.join(save_odir, "weight.pt")
    train_set = TrainDataset(mirna_arr, mrna_arr, label_arr)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    train_mirna = torch.FloatTensor(mirna_arr)
    train_mrna = torch.FloatTensor(mrna_arr)
    train_label = torch.LongTensor(label_arr)
    class_weight = torch.Tensor(compute_class_weight(class_weight='balanced',
                                                     classes=np.unique(label_arr),
                                                     y= label_arr)).to(device)
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
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, epochs), bar_format=bar_format) as tqdm_loader:
            for i, ((mirna, mrna), label) in enumerate(tqdm_loader):
                mirna, mrna, label = mirna.to(device, dtype=torch.float), mrna.to(device, dtype=torch.float), label.to(device)
                batch_pred = model((mirna, mrna))
                loss = criterion(batch_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i+1) == len(train_loader):
                    model.eval()
                    with torch.no_grad():
                        train_outputs = model.inference([train_mirna, train_mrna],
                                                        batch_size,
                                                        device)
                    _, train_pred = torch.max(train_outputs, 1)
                    train_loss = criterion(train_outputs, train_label)
                    train_probs = F.softmax(train_outputs, dim=1)[:, 1]
                    train_corrects = (train_pred == train_label).sum().item()
                    train_acc = train_corrects / len(train_pred)
                    train_roc_auc, train_pr_auc = calculation_scores(train_probs, train_label)
                    tqdm_loader.set_postfix(
                        dict(loss=train_loss.item(),
                             train_acc=train_acc))
                    log_fout.write(
                        "train_loss={:.6f}, train_acc={:.4f}, train_roc_auc={:.4f}, train_pr_auc={:.4f}".format(
                            train_loss, train_acc, train_roc_auc, train_pr_auc))
                else:
                    tqdm_loader.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), weight_file)
    log_fout.close()


def predict_result(mirna_fasta_ifile,
                   mrna_fasta_ifile,
                   query_file,
                   weight_file=None,
                   kmer=4,
                   batch_size=None,
                   seed_match='offset-9-mer-m7',
                   cts_size=30,
                   level='gene',
                   output_file=None,
                   device='cuda'):
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
    model.load_state_dict(torch.load(weight_file))
    dataset, neg_dataset = make_test_input_pair(mirna_fasta_ifile,
                                                mrna_fasta_ifile,
                                                query_file,
                                                cts_size=cts_size,
                                                kmer=kmer,
                                                seed_match=seed_match,
                                                header=True,
                                                train=True)
    mirna_arr = dataset['query_arr']
    mrna_arr = dataset['target_arr']
    mirna = torch.FloatTensor(mirna_arr)
    mrna = torch.FloatTensor(mrna_arr)
    device = torch.device(device)
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
    acc = (tp + tn)/len(results['LABEL'])
    sensitivity = tp / (tp + fn) # true positive rate
    specificity = tn / (tn + fp) # true negative rate
    roc_auc = metrics.roc_auc_score(results['LABEL'], results['PROBABILITY'])
    precision, recall, thresholds = metrics.precision_recall_curve(results['LABEL'], results['PROBABILITY'])
    pr_auc = metrics.auc(recall, precision)
    print("ACC: {:.4f}, ROC-AUC: {:.4f}, PR-AUC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}".format(acc, roc_auc,
                                                                                                          pr_auc,
                                                                                                          sensitivity,
                                                                                                          specificity))

def predict_result2(query_file,
                   weight_file=None,
                    kmer=4,
                    batch_size=128,
                   seed_match='offset-9-mer-m7',
                   cts_size=30,
                   level='gene',
                   output_file=None,
                   device='cuda'):
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
    model.load_state_dict(torch.load(weight_file))
    dataset, neg_dataset = make_test_input_pair2(query_file,
                                                 cts_size=cts_size,
                                                 kmer=kmer,
                                                 seed_match=seed_match,
                                                 header=True,
                                                 train=True)
    mirna_arr = dataset['query_arr']
    mrna_arr = dataset['target_arr']
    mirna = torch.FloatTensor(mirna_arr)
    mrna = torch.FloatTensor(mrna_arr)
    device = torch.device(device)
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
    acc = (tp + tn)/len(results['LABEL'])
    sensitivity = tp / (tp + fn)  # true positive rate
    specificity = tn / (tn + fp)  # true negative rate
    roc_auc = metrics.roc_auc_score(results['LABEL'], results['PROBABILITY'])
    precision, recall, thresholds = metrics.precision_recall_curve(results['LABEL'], results['PROBABILITY'])
    pr_auc = metrics.auc(recall, precision)
    print("ACC: {:.4f}, ROC-AUC: {:.4f}, PR-AUC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}".format(acc, roc_auc,
                                                                                                          pr_auc,
                                                                                                          sensitivity,
                                                                                                          specificity))


def predict_tcm(query_file,
                weight_file=None,
                kmer=4,
                batch_size=128,
                seed_match='offset-9-mer-m7',
                cts_size=30,
                level='gene',
                output_file=None,
                device='cuda'):
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
    model.load_state_dict(torch.load(weight_file))
    dataset, neg_dataset = make_test_input_pair2(query_file,
                                                 cts_size=cts_size,
                                                 kmer=kmer,
                                                 seed_match=seed_match,
                                                 header=True,
                                                 train=True)
    mirna_arr = dataset['query_arr']
    mrna_arr = dataset['target_arr']
    mirna = torch.FloatTensor(mirna_arr)
    mrna = torch.FloatTensor(mrna_arr)
    device = torch.device(device)
    model.to(device)
    with torch.no_grad():
        model.eval()
        outputs = model.inference([mirna, mrna], batch_size, device)
    _, predicts = torch.max(outputs.data, 1)
    probabilities = F.softmax(outputs, dim=1)

    y_probs = probabilities.cpu().numpy()[:, 1]
    y_predicts = predicts.cpu().numpy()

    postprocess_result(dataset, neg_dataset, y_probs, y_predicts,
                                 cts_size=cts_size, level=level, output_file=output_file)












