import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import regex
from utils.cgrModel2 import CGRModel
import pickle


def position_filter(positions_lst, cts_size, cutoff=3):
    positions_lst = sorted(positions_lst, reverse=True)
    new_positions = []
    if len(positions_lst) < cutoff + 1:
        return positions_lst
    for i in range(len(positions_lst) - cutoff):
        p = positions_lst[0]
        q = positions_lst[cutoff]
        positions_lst = positions_lst[1:]
        if p - q > cts_size + 5:
            new_positions.append(p)
    new_positions = new_positions + positions_lst
    return new_positions

def find_candidate(mirna_sequence, mrna_sequence, seed_match):
    positions = set()
    if seed_match == '10-mer-m6':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 6
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH
    elif seed_match == '10-mer-m7':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH
    elif seed_match == 'offset-9-mer-m7':
        SEED_START = 2
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END-SEED_START+1) - MIN_MATCH
    elif seed_match == 'strict':
        positions = find_strict_candidate(mirna_sequence, mrna_sequence)
        return positions
    else:
        raise ValueError(
            "seed_match expected 'strict', '10-mer-m6', '10-mer-m7', or 'offset-9-mer-m7', got '{}'".format(seed_match))
    seed = mirna_sequence[(SEED_START - 1):SEED_END]
    rc_seed = str(Seq(seed).reverse_complement().transcribe())
    match_iter = regex.finditer("({}){{e<={}}}".format(rc_seed, TOLERANCE), mrna_sequence)
    for match_index in match_iter:
        positions.add(match_index.end() + SEED_OFFSET)  # slice-stop indicies
    positions = list(positions)
    return positions

def find_strict_candidate(mirna_sequence, mrna_sequence):
    positions = set()
    SEED_TYPES = ['8-mer', '7-mer-m8', '7-mer-A1', '6-mer', '6-mer-A1', 'offset-7-mer', 'offset-6-mer']
    for seed_match in SEED_TYPES:
        if seed_match == '8-mer':
            SEED_START = 2
            SEED_END = 8
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '7-mer-m8':
            SEED_START = 1
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '7-mer-A1':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '6-mer':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 1
            seed = mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == '6mer-A1':
            SEED_START = 2
            SEED_END = 6
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START-1):SEED_END]
        elif seed_match == 'offset-7-mer':
            SEED_START = 3
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        rc_seed = str(Seq(seed).reverse_complement().transcribe())
        match_iter = regex.finditer(rc_seed, mrna_sequence)
        for match_index in match_iter:
            positions.add(match_index.end() + SEED_OFFSET)
    positions = list(positions)
    return positions

def get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match, up_stream=5, dn_stream=5, fix=False, cutoff=3):
    positions = find_candidate(mirna_sequence, mrna_sequence, seed_match)
    if fix:
        positions = position_filter(positions, cts_size=cts_size, cutoff=cutoff)
    candidates = []
    filtered_positions = []
    for i in positions:
        site_sequence = mrna_sequence[max(0, i-cts_size-up_stream):(i+dn_stream)]
        if len(site_sequence) < (cts_size+10):
            continue
        rc_site_sequence = str(Seq(site_sequence).reverse_complement().transcribe())
        candidates.append(rc_site_sequence) # miRNAs: 5'-ends to 3'-ends,  mRNAs: 3'-ends to 5'-ends
        filtered_positions.append(i)
    return candidates, filtered_positions

def make_pair(mirna_sequence, mrna_sequence, cts_size, seed_match, fix=False, cutoff=3):
    candidates, positions = get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match, fix=fix, cutoff=cutoff)
    mirna_querys = []
    mrna_targets = []
    if len(candidates) == 0:
        return (mirna_querys, mrna_targets, positions)
    else:
        for i in range(len(candidates)):
            mirna_querys.append(mirna_sequence)
            mrna_targets.append(candidates[i])
    return mirna_querys, mrna_targets, positions

def read_fasta(mirna_fasta_ifile, mrna_fasta_ifile):
    mirna_dict = dict()
    mrna_dict = dict()
    for sr in SeqIO.parse(mirna_fasta_ifile, 'fasta'):
        mirna_dict[sr.id] = str(sr.seq)
    for sr in SeqIO.parse(mrna_fasta_ifile, 'fasta'):
        mrna_dict[sr.id] = str(sr.seq)
    return mirna_dict, mrna_dict

def mirna_sequence_to_CGRmat(mirna_sequence, kmer=4, mode="RNA"):
    cgr = CGRModel(kmer=kmer, mode=mode)
    mat_1 = cgr.run(mirna_sequence)
    mat_2 = cgr.run(mirna_sequence[1:8])
    mat_3 = cgr.run(mirna_sequence, fill_type='idx', denominator=30)
    mat = np.stack([mat_1, mat_2, mat_3])
    return mat

def mrna_sequence_to_CGRmat(mrna_sequence, kmer=4, mode="RNA"):
    cgr = CGRModel(kmer=kmer, mode=mode)
    mat_1 = cgr.run(mrna_sequence)
    # mat_2 = cgr.run(mrna_sequence[5:-5])
    # mat = np.stack([mat_1, mat_2])
    return mat_1

def mirna_to_CGRmat(mirna_sequences, kmer=4, mode="RNA"):
    mat_lst = []
    for seq in mirna_sequences:
        mat = mirna_sequence_to_CGRmat(seq, kmer, mode)
        mat_lst.append(mat)
    mirna_arr = np.stack(mat_lst)
    return mirna_arr

def mrna_to_CGRmat(mrna_sequences, kmer=4, mode="RNA"):
    mat_lst = []
    for seq in mrna_sequences:
        mat = mrna_sequence_to_CGRmat(seq, kmer, mode)
        mat_lst.append(mat)
    mrna_arr = np.stack(mat_lst)
    return mrna_arr[:, np.newaxis, :, :]

def read_ground_truth(ground_truth_file, header=True, train=True):
    # input format: [MIRNA_ID, MRNA_ID, LABEL]
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')
    query_ids = np.asarray(records.iloc[:, 0].values)
    target_ids = np.asarray(records.iloc[:, 1].values)
    if train is True:
        labels = np.asarray(records.iloc[:, 2].values)
    else:
        labels = np.full((len(records),), fill_value=-1)
    return query_ids, target_ids, labels

def read_ground_truth2(ground_truth_file, header=True, train=True):
    # input format: [MIRNA_ID, MRNA_ID, LABEL]
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')
    query_ids = np.asarray(records.iloc[:, 0].values)
    target_ids = np.asarray(records.iloc[:, 1].values)
    if train is True:
        labels = np.asarray(records.iloc[:, 3].values)
    else:
        labels = np.full((len(records),), fill_value=-1)
    mirna_seqs = np.asarray(records.iloc[:, 4].values)
    mrna_seqs = np.asarray(records.iloc[:, 5].values)
    return query_ids, target_ids, labels, mirna_seqs, mrna_seqs

def read_ground_truth3(ground_truth_file, header=True):
    # input format: [MIRNA_ID, MRNA_ID, LABEL]
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')
    query_ids = np.asarray(records.iloc[:, 0].values)
    target_ids = np.asarray(records.iloc[:, 1].values)
    labels = np.full((len(records),), fill_value=-1)
    mirna_seqs = np.asarray(records.iloc[:, 2].values)
    mrna_seqs = np.asarray(records.iloc[:, 3].values)
    return query_ids, target_ids, labels, mirna_seqs, mrna_seqs

def make_train_input_pair(train_ifile, kmer=4, updn_stream=0):
    input_df = pd.read_csv(train_ifile, header=0, sep='\t')
    mirna_seqs = input_df['MIRNA_SEQ'].values.tolist()
    mrna_seqs = input_df['MRNA_SEQ'].values.tolist()
    mrna_seqs = [str(Seq(seq).reverse_complement().transcribe())[(5-updn_stream):(35+updn_stream)] for seq in mrna_seqs]
    mirna_arr = mirna_to_CGRmat(mirna_seqs, kmer=kmer, mode="RNA")
    mrna_arr = mrna_to_CGRmat(mrna_seqs, kmer=kmer, mode="RNA")
    label_arr = input_df['LABEL'].values.astype(int)
    return  mirna_arr, mrna_arr, label_arr


def make_test_input_pair(mirna_fasta_ifile,
                         mrna_fasta_ifile,
                         test_ifile,
                         cts_size=30,
                         kmer=4,
                         seed_match='offset-9-mer-m7',
                         header=True,
                         train=True,
                         fix=False,
                         cutoff=3):
    dataset = {
        'mirna_fasta_file': mirna_fasta_ifile,
        'mrna_fasta_file': mrna_fasta_ifile,
        'test_file': test_ifile,
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locations': [],
        'labels': []
    }
    neg_dataset = {
        'query_ids': [],
        'target_ids': [],
        'predicts': [],
        'labels': []
    }
    mirna_dict, mrna_dict = read_fasta(mirna_fasta_ifile, mrna_fasta_ifile)
    query_ids, target_ids, labels = read_ground_truth(test_ifile, header=header, train=train)
    for i in range(len(query_ids)):
        try:
            mirna_seq = mirna_dict[query_ids[i]]
        except KeyError:
            # print("MicroRNA {:s} has no sequences!!!".format(query_ids[i]))
            continue
        try:
            mrna_seq = mrna_dict[target_ids[i]]
        except KeyError:
            # print("Target {:s} has no sequence!!!".format(target_ids[i]))
            continue
        if len(mrna_seq) < (cts_size + 5 + 5):
            continue
        mrna_seq = str(Seq(mrna_seq).transcribe())
        query_seqs, target_seqs, locations = make_pair(mirna_seq, mrna_seq, cts_size=cts_size,
                                                       seed_match=seed_match, fix=fix, cutoff=cutoff)
        n_pairs = len(locations)
        if n_pairs > 0:
            queries = [query_ids[i] for n in range(n_pairs)]
            dataset['query_ids'].extend(queries)
            dataset['query_seqs'].extend(query_seqs)

            targets = [target_ids[i] for n in range(n_pairs)]
            dataset['target_ids'].extend(targets)
            dataset['target_seqs'].extend(target_seqs)
            dataset['target_locations'].extend(locations)

            dataset['labels'].extend([labels[i] for n in range(n_pairs)])
        else:
            neg_dataset['query_ids'].append(query_ids[i])
            neg_dataset['target_ids'].append(target_ids[i])
            neg_dataset['predicts'].append(0)
            neg_dataset['labels'].append(labels[i])
    dataset['query_arr'] = mirna_to_CGRmat(dataset['query_seqs'], kmer=kmer, mode='RNA')
    dataset['target_arr'] = mrna_to_CGRmat(dataset['target_seqs'], kmer=kmer, mode='RNA')
    neg_dataset['target_locations'] = [-1 for i in range(len(neg_dataset['query_ids']))]
    neg_dataset['probabilities'] = [0.0 for i in range(len(neg_dataset['query_ids']))]
    return dataset, neg_dataset

def make_test_input_pair2(test_ifile,
                          cts_size=30,
                          kmer=4,
                          seed_match='offset-9-mer-m7',
                          header=True,
                          train=True,
                          fix=False,
                          cutoff=3):
    dataset = {
        # 'mirna_fasta_file': mirna_fasta_ifile,
        # 'mrna_fasta_file': mrna_fasta_ifile,
        'test_file': test_ifile,
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locations': [],
        'labels': []
    }
    neg_dataset = {
        'query_ids': [],
        'target_ids': [],
        'predicts': [],
        'labels': []
    }
    # mirna_dict, mrna_dict = read_fasta(mirna_fasta_ifile, mrna_fasta_ifile)
    query_ids, target_ids, labels, mirna_seqs, mrna_seqs = read_ground_truth2(test_ifile, header=header, train=train)
    positions_num = 0
    for i in range(len(query_ids)):
        mirna_seq = mirna_seqs[i]
        mrna_seq = str(Seq(mrna_seqs[i]).transcribe())
        if len(mrna_seq) < (cts_size + 2*5):
            continue
        query_seqs, target_seqs, locations = make_pair(mirna_seq, mrna_seq, cts_size=cts_size,
                                                       seed_match=seed_match, fix=fix, cutoff=cutoff)
        n_pairs = len(locations)
        positions_num += n_pairs
        if n_pairs > 0:
            queries = [query_ids[i] for n in range(n_pairs)]
            dataset['query_ids'].extend(queries)
            dataset['query_seqs'].extend(query_seqs)

            targets = [target_ids[i] for n in range(n_pairs)]
            dataset['target_ids'].extend(targets)
            dataset['target_seqs'].extend(target_seqs)
            dataset['target_locations'].extend(locations)

            dataset['labels'].extend([labels[i] for n in range(n_pairs)])
        else:
            neg_dataset['query_ids'].append(query_ids[i])
            neg_dataset['target_ids'].append(target_ids[i])
            neg_dataset['predicts'].append(0)
            neg_dataset['labels'].append(labels[i])
    dataset['query_arr'] = mirna_to_CGRmat(dataset['query_seqs'], kmer=kmer, mode='RNA')
    dataset['target_arr'] = mrna_to_CGRmat(dataset['target_seqs'], kmer=kmer, mode='RNA')
    neg_dataset['target_locations'] = [-1 for i in range(len(neg_dataset['query_ids']))]
    neg_dataset['probabilities'] = [0.0 for i in range(len(neg_dataset['query_ids']))]
    print("average CTS number:", positions_num / len(query_ids))
    return dataset, neg_dataset

def make_test_input_pair3(test_ifile,
                          cts_size=30,
                          kmer=4,
                          seed_match='offset-9-mer-m7',
                          header=True,
                          fix=False,
                          cutoff=3):
    dataset = {
        # 'mirna_fasta_file': mirna_fasta_ifile,
        # 'mrna_fasta_file': mrna_fasta_ifile,
        'test_file': test_ifile,
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locations': [],
        'labels': []
    }
    neg_dataset = {
        'query_ids': [],
        'target_ids': [],
        'predicts': [],
        'labels': []
    }
    # mirna_dict, mrna_dict = read_fasta(mirna_fasta_ifile, mrna_fasta_ifile)
    query_ids, target_ids, labels, mirna_seqs, mrna_seqs = read_ground_truth3(test_ifile, header=header)
    for i in range(len(query_ids)):
        mirna_seq = mirna_seqs[i]
        mrna_seq = str(Seq(mrna_seqs[i]).transcribe())
        if len(mrna_seq) < (cts_size + 2*5):
            continue
        query_seqs, target_seqs, locations = make_pair(mirna_seq, mrna_seq, cts_size=cts_size,
                                                       seed_match=seed_match, fix=fix, cutoff=cutoff)
        n_pairs = len(locations)
        if n_pairs > 0:
            queries = [query_ids[i] for n in range(n_pairs)]
            dataset['query_ids'].extend(queries)
            dataset['query_seqs'].extend(query_seqs)

            targets = [target_ids[i] for n in range(n_pairs)]
            dataset['target_ids'].extend(targets)
            dataset['target_seqs'].extend(target_seqs)
            dataset['target_locations'].extend(locations)

            dataset['labels'].extend([labels[i] for n in range(n_pairs)])
        else:
            neg_dataset['query_ids'].append(query_ids[i])
            neg_dataset['target_ids'].append(target_ids[i])
            neg_dataset['predicts'].append(0)
            neg_dataset['labels'].append(labels[i])
    dataset['query_arr'] = mirna_to_CGRmat(dataset['query_seqs'], kmer=kmer, mode='RNA')
    dataset['target_arr'] = mrna_to_CGRmat(dataset['target_seqs'], kmer=kmer, mode='RNA')
    neg_dataset['target_locations'] = [-1 for i in range(len(neg_dataset['query_ids']))]
    neg_dataset['probabilities'] = [0.0 for i in range(len(neg_dataset['query_ids']))]
    return dataset, neg_dataset

def postprocess_result(dataset,
                       neg_dataset,
                       probabilities,
                       predicts,
                       cts_size,
                       output_file=None,
                       level='gene'):
    query_ids = np.append(dataset['query_ids'], neg_dataset['query_ids'])
    target_ids = np.append(dataset['target_ids'], neg_dataset['target_ids'])
    target_locs = np.append(dataset['target_locations'], neg_dataset['target_locations'])
    probabilities = np.append(probabilities, neg_dataset['probabilities'])
    predicts = np.append(predicts, neg_dataset['predicts'])
    labels = np.append(dataset['labels'], neg_dataset['labels'])

    # output format: [QUERY, TARGET, LOCATION, PROBABILITY]
    output_df = pd.DataFrame(columns=['MIRNA_ID', 'MRNA_ID', 'LOCATION', 'PROBABILITY'])
    output_df['MIRNA_ID'] = query_ids
    output_df['MRNA_ID'] = target_ids
    output_df['LOCATION'] = np.array(
        ["{},{}".format(max(1, l - cts_size + 1), l) if l != -1 else "-1,-1" for l in target_locs])
    output_df['PROBABILITY'] = probabilities
    output_df['PREDICT'] = predicts
    output_df['LABEL'] = labels

    output_df = output_df.sort_values(by=['PROBABILITY', 'MIRNA_ID', 'MRNA_ID'], ascending=[False, True, True])
    # print(output_df.shape)
    unique_output_df = output_df.sort_values(by=['PROBABILITY', 'MIRNA_ID', 'MRNA_ID'],
                                             ascending=[False, True, True]).drop_duplicates(
        subset=['MIRNA_ID', 'MRNA_ID'], keep='first')
    if level == 'site':
        if output_file is not None:
            output_df.to_csv(output_file, index=False, sep='\t')
        return output_df
    elif level == 'gene':
        if output_file is not None:
            unique_output_df.to_csv(output_file, index=False, sep='\t')
        return unique_output_df
    else:
        raise ValueError("Parameter level expected 'site' or 'gene', got '{}'".format(mode))




def load_train(ifile):
    with open(ifile, 'rb') as fin:
        train_dat_dict = pickle.load(fin)
    return train_dat_dict












