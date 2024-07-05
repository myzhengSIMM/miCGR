
import numpy as np
import re

# 在python3中, 默认继承object
# source: https://github.com/dognjen/chaos-game-representation
class CGRModel:
    def __init__(self, kmer, mode="DNA"):
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

    def count(self, sequence, sub):
        """ Method counts substrings in a given sequence.
        Moving through the sequence by one nucleotide."""
        ss = re.compile('(?={})'.format(sub))
        return len(re.findall(ss, sequence))

    def location(self, sequence, sub, denominator=30):
        idx_lst = []
        idx = sequence.find(sub)
        while idx != -1:
            idx_lst.append(idx)
            idx = sequence.find(sub, idx+1)
        if len(idx_lst) > 0:
            return denominator - sum(idx_lst)/len(idx_lst)
        else:
            return 0

    def k_mer_probability(self, sequence, bases, emphasized_seqs = [], fold=0):
        """ Method calculates the probability of k-mer in a given sequence. """
        probabilities = dict()
        n = 0
        for s in emphasized_seqs:
            n += self.count(sequence, s) * (fold-1)
        for b in bases:
            if b in emphasized_seqs:
                probabilities[b] = self.count(sequence, b)*fold / float(len(sequence) - len(b) + 1 + n)
            else:
                probabilities[b] = self.count(sequence, b) / float(len(sequence) - len(b) + 1 + n)
        return probabilities

    def k_mer_location(self, sequence, bases, denominator=30):
        locations = dict()
        for b in bases:
            locations[b] = self.location(sequence, b, denominator) / float(denominator)
        return locations

    def k_mer_mapping(self, bases):
        mappings = dict()
        for b in bases:
            mappings[b] = b
        return mappings

    def fill_matrix(self, bases, depth, row, column, nuk):
        """ This recursion fills the matrix of nucleotides with probabilities for chaos game representation.
         Nucleotide A represents upper left corner, nucleotide C represents bottom left corner, G represents bottom right,
         while T/U represents upper right corner of the image. """

        if depth == 0:
            self.mat[column[0]][row[0]] = self.ver[nuk]
            return []
        else:
            for b in bases:
                if b in ['A', 'C']:
                    r = row[:len(row) // 2]
                else:
                    r = row[len(row) // 2:]
                if b in ['A', 'T', 'U']:
                    c = column[:len(column) // 2]
                else:
                    c = column[len(column) // 2:]
                self.fill_matrix(bases, depth - 1, r, c, nuk + b)
            return []

    def get_mapping_matrix(self):
        self.ver = self.k_mer_mapping(self.combinations)
        self.mat = np.array([['0'*self.k for i in range(self.s)] for j in range(self.s)])
        self.fill_matrix(self.bases, self.k,
                         [int(i) for i in range(self.s)],
                         [int(j) for j in range(self.s)],
                         '')
        return self.mat

    def run(self, sequence, emphasized_seqs = [], fold=0, fill_type="prob", denominator=30):
        # convert to upper case and remove unknown nucleotides
        sequence = sequence.upper().replace("N", "")
        # initialize a matrix
        self.mat = np.array([[0.0 for i in range(self.s)] for j in range(self.s)])
        # Calculate the probability (k-mer) of all combinations in a given sequence
        if fill_type == "prob":
            self.ver = self.k_mer_probability(sequence, self.combinations,
                                              emphasized_seqs = emphasized_seqs, fold=fold) # dict
        elif fill_type == "idx":
            self.ver = self.k_mer_location(sequence, self.combinations,
                                           denominator=denominator)
        self.fill_matrix(self.bases, self.k,
                                            [int(i) for i in range(self.s)],
                                            [int(j) for j in range(self.s)],
                                            '')
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






