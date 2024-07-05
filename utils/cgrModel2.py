
import numpy as np
import re

# 在python3中, 默认继承object
# source: https://github.com/dognjen/chaos-game-representation
class CGRModel:
    def __init__(self, kmer, mode):
        self.k = kmer # length  of nucleotides
        self.s = 2**kmer
        if mode == "DNA":
            self.bases = ["A", "C", "G", "T"]
        elif mode == "RNA":
            self.bases = ["A", "C", "G", "U"]
        # Create all possible k combinations of base nucleotides
        self.combinations = list(self.cartesian([self.bases] * self.k))

    def cartesian(self, nucleotides):
        """ Create a cartesian product from variable nucleotides. """
        if not nucleotides:
            yield ""
        else:
            for a in nucleotides[0]:
                for b in self.cartesian(nucleotides[1:]):
                    yield a + b

    def uniq_sub_seq(self, sequence, sub_len):
        oset = set()
        for i in range(len(sequence)-sub_len):
            oset.add(sequence[i:(i+sub_len)])
        return oset

    def count(self, sequence, sub_seq):
        """ Method counts substrings in a given sequence.
        Moving through the sequence by one nucleotide."""
        ss = re.compile('(?={})'.format(sub_seq))
        return len(re.findall(ss, sequence))

    def location(self, sequence, sub_seq, denominator=30):
        idx_lst = []
        idx = sequence.find(sub_seq)
        while idx != -1:
            idx_lst.append(idx)
            idx = sequence.find(sub_seq, idx+1)
        if len(idx_lst) > 0:
            return denominator - sum(idx_lst)/len(idx_lst)
        else:
            return 0

    def k_mer_probability(self, sequence, emphasized_seqs = [], fold=0):
        """ Method calculates the probability of k-mer in a given sequence. """
        probabilities = dict()
        n = 0
        sub_seqs = self.uniq_sub_seq(sequence, self.k)
        for s in emphasized_seqs:
            n += self.count(sequence, s) * (fold-1)
        for sub_seq in sub_seqs:
            if sub_seq in emphasized_seqs:
                probabilities[sub_seq] = self.count(sequence, sub_seq)*fold / float(len(sequence) - len(sub_seq) + 1 + n)
            else:
                probabilities[sub_seq] = self.count(sequence, sub_seq) / float(len(sequence) - len(sub_seq) + 1 + n)
        return probabilities

    def k_mer_location(self, sequence, denominator=30):
        locations = dict()
        sub_seqs = self.uniq_sub_seq(sequence, self.k)
        for sub_seq in sub_seqs:
            locations[sub_seq] = self.location(sequence, sub_seq, denominator) / float(denominator)
        return locations

    def k_mer_mapping(self, bases):
        mappings = dict()
        for b in bases:
            mappings[b] = b
        return mappings

    def fill_matrix_by_sub_seq(self, sub_seq, row, column):
        """ This recursion fills the matrix of nucleotides with probabilities for chaos game representation.
         Nucleotide A represents upper left corner, nucleotide C represents bottom left corner, G represents bottom right,
         while T/U represents upper right corner of the image. """
        for b in sub_seq:
            if b in ['A', 'C']:
                row = row[:len(row) // 2]
            else:
                row = row[len(row) // 2:]
            if b in ['A', 'T', 'U']:
                column = column[:len(column) // 2]
            else:
                column = column[len(column) // 2:]
        self.mat[column[0]][row[0]] = self.ver[sub_seq]

    def fill_matrix(self, fill_dict):
        for sub_seq in fill_dict.keys():
            self.fill_matrix_by_sub_seq(sub_seq,
                                        [int(i) for i in range(self.s)],
                                        [int(j) for j in range(self.s)])

    def get_mapping_matrix(self):
        self.ver = self.k_mer_mapping(self.combinations)
        self.mat = np.array([['0'*self.k for i in range(self.s)] for j in range(self.s)])
        self.fill_matrix(self.ver)
        return self.mat

    def run(self, sequence, emphasized_seqs = [], fold=0, fill_type="prob", denominator=30):
        # convert to upper case and remove unknown nucleotides
        sequence = sequence.upper().replace("N", "")
        # initialize a matrix
        self.mat = np.array([[0.0 for i in range(self.s)] for j in range(self.s)])
        # Calculate the probability (k-mer) of all combinations in a given sequence
        if fill_type == "prob":
            self.ver = self.k_mer_probability(sequence,
                                              emphasized_seqs = emphasized_seqs, fold=fold) # dict
        elif fill_type == "idx":
            self.ver = self.k_mer_location(sequence,
                                           denominator=denominator)
        self.fill_matrix(self.ver)
        return self.mat

    def plot(self, mat):
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        s = mat.shape[0]
        pylab.imshow(mat, extent=[0, s, 0, s], interpolation='nearest', cmap=cm.gray)
        x_os = [a + 0.4 for a in range(s)]
        y_os = [a + 0.4 for a in range(s)]
        label_x = [str(i) for i in range(s)]
        label_y = [str(i) for i in range(s - 1, -1, -1)]
        pylab.xticks(x_os, label_x)
        pylab.yticks(y_os, label_y)
        pylab.show()






