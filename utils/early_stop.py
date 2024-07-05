import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,
                 patience=7,
                 verbose=False,
                 delta=0,
                 param_name='val_loss',
                 mode='min',
                 path='checkpoint.pt',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): The monitoring direction, one in `{"min", "max"}`, default: 'min'
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stopping_flag = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.param_name = param_name
        self.saving_flag = False
        if mode == 'min':
            self.monitor_op = np.less
            self.delta *= -1
            self.last_best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.delta *= 1
            self.last_best = - np.Inf
        else:
            raise ValueError("'mode' expected 'min' or 'max', got '{}'".format(self.mode))
    def __call__(self, monitor_val, model):
        if self.best_score is None:
            self.best_score = monitor_val
            self.save_checkpoint(monitor_val, model)
            self.saving_flag = True
        else:
            if self.monitor_op(monitor_val-self.delta, self.best_score):
                self.best_score = monitor_val
                self.save_checkpoint(monitor_val, model)
                self.counter = 0
                self.saving_flag = True
            else:
                self.saving_flag = False
                self.counter += 1
                # self.trace_func('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    self.stopping_flag = True

    def save_checkpoint(self, monitor_val, model):
        '''Saves model when monitor_val improves.'''
        if self.verbose:
            self.trace_func('`{}` decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                self.param_name, self.last_best, monitor_val))
        torch.save(model.state_dict(), self.path)
        self.last_best = monitor_val
