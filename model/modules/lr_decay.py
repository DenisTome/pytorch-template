# -*- coding: utf-8 -*-
"""
Created on Jun 04 13:59 2018

@author: Denis Tome'

Implementation of learning rate decay

"""

__all__ = [
    'LRDecay'
]


class LRDecay(object):
    """Exponential learning rate decay"""

    def __init__(self, optimizer, lr, decay_rate, decay_steps, mode='exp'):
        """
        Implementation of learning rate decay
        :param optimizer: optimizer from torch.optim
        :param lr: initial learning rate
        :param decay_rate: the decay rate
        :param decay_steps: used for decay computation
        :param mode: decay mode (only 'exp' supported so far)
        """
        super(LRDecay, self).__init__()

        # only version supported is exponential decay
        assert (mode in ['exp'])
        assert (lr > 0.0)
        assert (decay_rate > 0.0)
        assert (decay_steps > 0)
        assert (type(decay_steps) == int)

        self.optimizer = optimizer
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_step = decay_steps
        self.mode = mode

    def _exponential_decay(self, iter):
        return self.lr * self.decay_rate ** (iter / self.decay_step)

    def update_lr(self, global_step):
        """
        Update learning rate depending on the global_step
        :param global_step
        """

        new_lr = None
        if self.mode is 'exp':
            new_lr = self._exponential_decay(global_step)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
