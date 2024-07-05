import torch
import numpy as np
import pandas as pd
from utils.sequence import make_train_input_pair, mirna_to_CGRmat, mrna_to_CGRmat
from Bio.Seq import Seq



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, mirna_arr, mrna_arr, label_arr):
        self.mirna_arr = mirna_arr
        self.mrna_arr = mrna_arr
        self.label_arr = label_arr

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mirna_arr = self.mirna_arr[idx]
        mrna_arr = self.mrna_arr[idx]
        label_arr = self.label_arr[idx]
        return (mirna_arr, mrna_arr), label_arr

    def __len__(self):
        return len(self.label_arr)




