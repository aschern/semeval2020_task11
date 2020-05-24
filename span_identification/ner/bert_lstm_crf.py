# -*- coding: utf-8 -*-
"""
based on
@File: bert_lstm_crf.py
@Copyright: 2019 Michael Zhu
@License：the Apache License, Version 2.0
@Author：Michael Zhu
"""

# coding=utf-8
# coding=utf-8
import copy
from typing import cast, List
import numpy as np

import torch.nn as nn

from torch.autograd import Variable
import torch

from .conditional_random_field import ConditionalRandomField, allowed_transitions


class BertLstmCrf(nn.Module):
    """
    bert_lstm_crf model
    """

    def __init__(self, bert_model,
                 num_labels=9,
                 embedding_dim=512,
                 hidden_dim=512,
                 rnn_layers=1,
                 rnn_dropout=0.1,
                 output_dropout=0.1,
                 use_cuda=False):
        super(BertLstmCrf, self).__init__()
        self.bert_encoder = bert_model

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers

        self.lstm = None
        if rnn_layers > 0:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=rnn_layers,
                bidirectional=True,
                dropout=rnn_dropout,
                batch_first=True
            )

        # self.crf = CRF(
        #     target_size=num_labels,
        #     average_batch=True,
        #     use_cuda=use_cuda
        # )

        # TODO: add contraints
        constraints = allowed_transitions('BIO', dict(enumerate(["O", "B", "I"])))
        include_start_end_transitions = True
        self.crf = ConditionalRandomField(
            num_labels,
            constraints,
            include_start_end_transitions=include_start_end_transitions
        )

        self.liner = nn.Linear(hidden_dim * 2, num_labels)
        self.num_labels = num_labels

        self.output_dropout = nn.Dropout(p=output_dropout)

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))
    
    def clear_subtokens(self, logits, labels, mask):
        clear_labels = torch.zeros_like(labels)
        clear_logits = torch.zeros_like(logits)
        clear_mask = torch.zeros_like(mask)

        for i in range(len(labels)):
            assert (mask[i][labels[i] != -100] == 1).all()
            cor = labels[i][labels[i] != -100]
            clear_labels[i][:len(cor)] = cor
            clear_logits[i][:len(cor)] = logits[i][labels[i] != - 100]
            clear_mask[i][:len(cor)] = 1
        return clear_logits, clear_labels, clear_mask

    def forward(self, **kwargs):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''

        kwargs_copy = copy.deepcopy(kwargs)
        if "labels" in kwargs_copy:
            kwargs_copy.pop("labels")

        batch_size = kwargs["input_ids"].size(0)
        seq_length = kwargs["input_ids"].size(1)

        bert_outputs = self.bert_encoder(
            **kwargs
        )
        sequence_output = bert_outputs[1]

        if self.lstm is not None:
            hidden = self.rand_init_hidden(batch_size)
            if kwargs["input_ids"].is_cuda:
                hidden = (i.cuda() for i in hidden)
            sequence_output, hidden = self.lstm(sequence_output, hidden)
            sequence_output = sequence_output.contiguous().view(-1, self.hidden_dim * 2)
            sequence_output = self.output_dropout(sequence_output)
            
            sequence_output = self.liner(sequence_output)

        #out = self.liner(sequence_output)
        out = sequence_output
        logits = out.contiguous().view(batch_size, seq_length, -1)
        
        clear_logits, clear_labels, clear_mask = self.clear_subtokens(logits, kwargs['labels'], kwargs["attention_mask"])
        
        """
        best_paths = self.crf.viterbi_tags(
            logits,
            kwargs["attention_mask"].long(),
            top_k=1
        )
        """
        best_paths = self.crf.viterbi_tags(
            clear_logits,
            clear_mask.long(),
            top_k=1
        )
        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
        
        if kwargs.get("labels") is not None:
            labels = kwargs.get("labels").cpu()
            #log_likelihood = self.crf(logits, kwargs.get("labels"), kwargs["attention_mask"])
            log_likelihood = self.crf(clear_logits, clear_labels, clear_mask)
            loss = -log_likelihood
            correct_predicted_tags = np.zeros_like(labels)
            for i in range(len(labels)):
                correct_predicted_tags[i][labels[i] != -100] = predicted_tags[i]
            return (loss, logits, list(correct_predicted_tags))

        return (None, logits, predicted_tags)


if __name__ == "__main__":
    pass
