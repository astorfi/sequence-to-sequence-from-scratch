from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import argparse

SOS_token = 1
EOS_token = 2




def SentenceFromTensor_(lang, tensor):
    indexes = tensor.squeeze()
    indexes = indexes.tolist()
    return [lang.index2word[index] for index in indexes]


# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)


def tensorsFromPair(pair, input_lang, output_lang, max_input_length):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])

    with torch.no_grad():

        # Pad buttom with zeros for getting a fixed length.
        pad_input = nn.ConstantPad1d((0, max_input_length - input_tensor.shape[1]), 0)
        pad_target = nn.ConstantPad1d((0, max_input_length - target_tensor.shape[1]), 0)

        # Padding operation
        input_tensor_padded = pad_input(input_tensor)
        target_tensor_padded = pad_target(target_tensor)

    # The "pad_sequence" function is used to pad the shorter sentence to make the tensors of equal size
    from torch.nn.utils.rnn import pad_sequence
    pair_tensor = pad_sequence([input_tensor_padded, target_tensor_padded], batch_first=False, padding_value=0)

    return pair_tensor

######################################################################
def reformat_tensor_(tensor):
    tensor = tensor.transpose(0, 2, 1)
    tensor = tensor.squeeze()
    return tensor[tensor != -1].view(-1, 1)

def reformat_tensor_mask(tensor):
    tensor = tensor.squeeze(dim=1)
    tensor = tensor.transpose(1,0)
    mask = tensor != 0
    return tensor, mask