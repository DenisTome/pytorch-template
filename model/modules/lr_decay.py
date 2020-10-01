# -*- coding: utf-8 -*-
"""
Implementation of learning rate decay


@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__all__ = [
    'LRDecay'
]


class LRDecay:
    """Exponential learning rate decay"""

    def __init__(self, optimizer, lr, decay_rate, decay_steps, mode='exp'):
        """Implementation of leaning rate decay

        Arguments:
            optimizer {Optimizer} -- torch optimizer
            lr {float} -- initial learning rate
            decay_rate {float} -- decay rate
            decay_steps {int} -- iteration after applying the decay

        Keyword Arguments:
            mode {str} -- behaviour of decay (default: {'exp'})
        """

        super(LRDecay, self).__init__()

        # only version supported is exponential decay
        assert mode == 'exp'
        assert lr > 0.0
        assert decay_rate > 0.0
        assert decay_steps > 0
        assert isinstance(decay_steps, int)

        self.optimizer = optimizer
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_step = decay_steps
        self.mode = mode

    def _exponential_decay(self, iteration):
        """Apply decay"""
        return self.lr * self.decay_rate ** (iteration / self.decay_step)

    def update_lr(self, global_step):
        """Update based on global step

        Arguments:
            global_step {int} -- global step
        """

        new_lr = None
        if self.mode == 'exp':
            new_lr = self._exponential_decay(global_step)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
