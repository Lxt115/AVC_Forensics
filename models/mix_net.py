#!/usr/bin/env python
# coding=utf-8

"""
@author: Richard Huang
@license: WHU
@contact: 2539444133@qq.com
@file: mix_net.py
@date: 22/05/15 21:46
@desc: 
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


class MixModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MixModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        h0 = h0.cuda()
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # predictions based on every time step
        return out