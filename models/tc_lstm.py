# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class TC_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        self.opt = opt
        super(TC_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        x_l, x_r, target = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1).view(len(target),1).float().repeat(1,self.opt.embed_dim)
        target_embedding = torch.sum(self.embed(target),dim=1)/target_len
        target_embedding = target_embedding.unsqueeze(1).repeat(1,x_l.size()[-1],1)
        x_l, x_r = self.embed(x_l)+target_embedding, self.embed(x_r)+target_embedding

        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out
