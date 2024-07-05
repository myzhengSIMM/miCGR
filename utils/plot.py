import shap
import regex
from itertools import islice
from Bio.Seq import Seq
import pickle
from utils.cgrModel2 import CGRModel
import numpy as np

def mirna_sequence_to_CGRmat(mirna_sequence, kmer=6, mode="RNA"):
    cgr = CGRModel(kmer=kmer, mode=mode)
    mat_1 = cgr.run(mirna_sequence)
    mat_2 = cgr.run(mirna_sequence[1:8])
    mat_3 = cgr.run(mirna_sequence, fill_type='idx', denominator=30)
    mat = np.stack([mat_1, mat_2, mat_3])
    return mat

def mrna_sequence_to_CGRmat(mrna_sequence, kmer=6, mode="RNA"):
    cgr = CGRModel(kmer=kmer, mode=mode)
    mat_1 = cgr.run(mrna_sequence)
    # mat_2 = cgr.run(mrna_sequence[5:-5])
    # mat = np.stack([mat_1, mat_2])
    return mat_1

def select_pos_sample(nSample=10, kmer=6):
    olst = list()
    n = 0
    with open('data/train/train_set_shuffled.csv', 'r') as fin:
        for line in islice(fin, 1, None):
            lst = line.strip('\n').split('\t')
            if lst[-1] == '0': continue
            if 'LLL' in lst[3]: continue
            mirna_seq = lst[3]
            mrna_seq = str(Seq(lst[4]).reverse_complement().transcribe())
            match_lst=list(regex.finditer("({}){{e<={}}}".format(mirna_seq[1:10], 2), mrna_seq[6:15]))
            if match_lst == []: continue
            n += 1
            mirna_mat = mirna_sequence_to_CGRmat(mirna_seq, kmer=kmer)
            mrna_mat = mrna_sequence_to_CGRmat(mrna_seq, kmer=kmer)
            olst.append({"mirna_id": lst[0],
                        "mrna_id": lst[2],
                        "mirna_seq": mirna_seq,
                        "mrna_seq": mrna_seq,
                         "mirna_mat": mirna_mat,
                         "mrna_mat": mrna_mat})
            if n >= nSample: break
    return olst

def select_neg_sample(nSample=10, kmer=6):
    olst = list()
    n = 0
    with open('data/train/train_set_shuffled.csv', 'r') as fin:
        for line in islice(fin, 1, None):
            lst = line.strip('\n').split('\t')
            if lst[-1] == '1': continue
            if 'LLL' in lst[3]: continue
            mirna_seq = lst[3]
            mrna_seq = str(Seq(lst[4]).reverse_complement().transcribe())
            match_lst=list(regex.finditer("({}){{e<={}}}".format(mirna_seq[1:10], 2), mrna_seq[6:15]))
            if match_lst == []: continue
            n += 1
            mirna_mat = mirna_sequence_to_CGRmat(mirna_seq, kmer=kmer)
            mrna_mat = mrna_sequence_to_CGRmat(mrna_seq, kmer=kmer)
            olst.append({"mirna_id": lst[0],
                         "mrna_id": lst[2],
                         "mirna_seq": mirna_seq,
                         "mrna_seq": mrna_seq,
                         "mirna_mat": mirna_mat,
                         "mrna_mat": mrna_mat})
            if n >= nSample: break
    return olst

def construct_plot_dat(nSample=10, kmer=6):
    pos_lst = select_pos_sample(nSample=nSample, kmer=kmer)
    neg_lst = select_neg_sample(nSample=nSample, kmer=kmer)
    odict = {"pos": pos_lst, "neg": neg_lst}
    with open("data/plot/dat_k{:d}.pkl".format(kmer), 'wb') as fout:
        pickle.dump(odict, fout)
