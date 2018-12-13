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
from _utils.transformer import *


# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")


# Device parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################
### PREPROCESSING ######
########################

# Please refer to https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
######################################################################
# We have to define word indexing for further processing.
# word2index: Word to its associated index
# index2word: Index to the associated word.

SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<pad>", SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub("[.!?]", '', s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.

def readLangs(lang1, lang2, auto_encoder=False, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % ('eng', 'fra'), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Autoencoder have the same data as the output
    if auto_encoder:
        pairs = [[pair[0], pair[0]] for pair in pairs]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p, max_input_length):
    return len(p[0].split(' ')) < max_input_length and \
           len(p[1].split(' ')) < max_input_length and \
           p[1].startswith(eng_prefixes)

def filterPairs(pairs, max_input_length):
    pairs = [pair for pair in pairs if filterPair(pair, max_input_length)]
    return pairs

######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, max_input_length, auto_encoder=False, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, auto_encoder, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_input_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



########################
### Dataset Class ######
########################


class Dataset():
    """dataset object"""

    def __init__(self, phase, num_embeddings=None, max_input_length=None, transform=None, auto_encoder=False):
        """
        The initialization of the dataset object.
        :param phase: train/test.
        :param num_embeddings: The embedding dimentionality.
        :param max_input_length: The maximum enforced length of the sentences.
        :param transform: Post processing if necessary.
        :param auto_encoder: If we are training an autoencoder or not.
        """
        if auto_encoder:
            lang_in = 'eng'
            lang_out = 'eng'
        else:
            lang_in = 'eng'
            lang_out = 'fra'
        # Skip and eliminate the sentences with a length larger than max_input_length!
        input_lang, output_lang, pairs = prepareData(lang_in, lang_out, max_input_length, auto_encoder=auto_encoder, reverse=True)
        print(random.choice(pairs))

        # Randomize list
        random.shuffle(pairs)

        if phase == 'train':
            selected_pairs = pairs[0:int(0.8 * len(pairs))]
        else:
            selected_pairs = pairs[int(0.8 * len(pairs)):]

        # Getting the tensors
        selected_pairs_tensors = [tensorsFromPair(selected_pairs[i], input_lang, output_lang, max_input_length)
                     for i in range(len(selected_pairs))]

        self.transform = transform
        self.num_embeddings = num_embeddings
        self.max_input_length = max_input_length
        self.data = selected_pairs_tensors
        self.input_lang = input_lang
        self.output_lang = output_lang

    def langs(self):
        return self.input_lang, self.output_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # A tuple which represent a pair
        pair = self.data[idx]

        # Define the sample dictionary
        sample = {'sentence': pair}

        if self.transform:
            sample = self.transform(sample)

        return sample



# ######################################
# #Uncomment for testing dataset class #
# ######################################
#
# # Create training data object
# trainset = Dataset(phase='train', max_input_length=10)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                           shuffle=True, num_workers=1, pin_memory=False)
#
# dataiter = iter(trainloader)
# item = dataiter.next()
# print(item['sentence'].shape)


# sentences = item
# print("Shape of a sample mini-batch: ", sentences.shape)

# # ###########################
# # Loop #
# # ###########################
#
# for i in range(len(trainset)):
#     sample = trainset[i]
#
#     print(i, sample['sentence'].shape)
#     print(sample['sentence'][0])
#
#     break