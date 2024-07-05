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
from utils.sequence import load_train, make_test_input_pair, postprocess_result, make_test_input_pair3
from utils.normalizers import ChannelNormalizer
from utils.early_stop import EarlyStopping
import os
import pickle
import sys


def predict(ifile, gpu_n):
    print("Loading model...")
    weight_file = "./model/model_weights.pt"
    level = "gene"
    kmer = 6
    cuda = "cuda:{:d}".format(gpu_n)
    dropout = 0.5
    model = NSWSSC_k6(dropout=dropout)
    cts_size = 30
    seed_match = sem
    batch_size = 4096
    model.load_state_dict(torch.load(weight_file))
    device = torch.device(cuda)
    model.to(device)
    query_file = os.path.join(ifile)
    output_file = os.path.join(os.path.splitext(ifile)[0] +'_{:s}'.format(seed_match) +'_res.txt')
    print("Constructing input...")
    dataset, neg_dataset = make_test_input_pair3(query_file,
                                                 cts_size=cts_size,
                                                 kmer=kmer,
                                                 seed_match=seed_match,
                                                 header=True,
                                                 fix=True,
                                                 cutoff=1)
    mirna_arr = dataset['query_arr']
    mrna_arr = dataset['target_arr']
    mirna = torch.FloatTensor(mirna_arr)
    mrna = torch.FloatTensor(mrna_arr)
    print("Predicting...")
    with torch.no_grad():
        model.eval()
        outputs = model.inference([mirna, mrna], batch_size, device)
    _, predicts = torch.max(outputs.data, 1)
    probabilities = F.softmax(outputs, dim=1)
    y_probs = probabilities.cpu().numpy()[:, 1]
    y_predicts = predicts.cpu().numpy()

    postprocess_result(dataset, neg_dataset, y_probs, y_predicts,
                   cts_size=cts_size, level=level, output_file=output_file)
if __name__=="__main__":
    ifile = sys.argv[1]
    gpu_n = int(sys.argv[2])
    sem = sys.argv[3]
    predict(ifile, gpu_n)

