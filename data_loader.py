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


# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")

# Parser
parser = argparse.ArgumentParser(description='Creating Classifier')

###############
# Model Flags #
###############
parser.add_argument('--auto_encoder', default=False, type=str2bool, help='Use auto-encoder model')

# Add all arguments to parser
args = parser.parse_args()

# Device parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################
### PREPROCESSING ######
########################

######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if args.auto_encoder:
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

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
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
    # return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)


def tensorsFromPair(pair, input_lang, output_lang, max_input_length):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])

    with torch.no_grad():

        # Pad buttom with zeros for getting a fixed length.
        pad_input = nn.ConstantPad1d((0, max_input_length - input_tensor.shape[1]),-1)
        pad_target = nn.ConstantPad1d((0, max_input_length - target_tensor.shape[1]), -1)

        # Padding operation
        input_tensor_padded = pad_input(input_tensor)
        target_tensor_padded = pad_target(target_tensor)

    # The "pad_sequence" function is used to pad the shorter sentence to make the tensors of equal size
    from torch.nn.utils.rnn import pad_sequence
    pair_tensor = pad_sequence([input_tensor_padded, target_tensor_padded], batch_first=False, padding_value=-1)

    return pair_tensor


########################
### Dataset Class ######
########################


class Dataset():
    """Face Landmarks dataset."""

    def __init__(self, phase, num_embeddings=None, max_input_length=None, transform=None):
        """
        Args:
            split (string): Here we define the split. The choices are: 'trnid', 'tstid' and 'valid' based on flower dataset.
            mat_file_data_split (string): Path to the mat file with indexes for the specific split.
            mat_file_label (string): Path to the labels based on the file index.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.num_embeddings = num_embeddings
        self.max_input_length = max_input_length

        # Skip and eliminate the sentences with a length larger than max_input_length!
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
        print(random.choice(pairs))

        if phase == 'train':
            selected_pairs = pairs[0:int(0.8 * len(pairs))]
        else:
            selected_pairs = pairs[int(0.8 * len(pairs)):]

        # Getting the tensors
        selected_pairs_tensors = [tensorsFromPair(selected_pairs[i], input_lang, output_lang, self.max_input_length)
                     for i in range(len(selected_pairs))]

        self.data = selected_pairs_tensors

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



#######################################
# Uncomment for testing dataset class #
#######################################

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