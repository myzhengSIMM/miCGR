import pickle

class ChannelNormalizer(object):
    def __init__(self, n_channels=3):
        self.n_channels = n_channels
        self.means = [0] * self.n_channels
        self.stds = [0] * self.n_channels

    def fit(self, nd_arr):
        for i in range(self.n_channels):
            self.means[i] = nd_arr[:, i, :, :].reshape(nd_arr.shape[0], -1).mean(1).mean()
            self.stds[i] = nd_arr[:, i, :, :].reshape(nd_arr.shape[0], -1).std(1).mean()

    def transform(self, nd_arr):
        for i in range(self.n_channels):
            nd_arr[:, i, :, :] = (nd_arr[:, i, :, :]-self.means[i]) / self.stds[i]
        return nd_arr

    def fit_transform(self, nd_arr):
        self.fit(nd_arr)
        out_nd_arr = self.transform(nd_arr)
        return out_nd_arr

    def load(self, ifile):
        with open(ifile, 'rb') as fin:
            idict = pickle.load(fin)
        self.n_channels = len(idict["means"])
        self.means = idict["means"]
        self.stds = idict["stds"]

    def save(self, ofile):
        odict = {"means": self.means,
                 "stds": self.stds}
        with open(ofile, 'wb') as fout:
            pickle.dump(odict, fout)




