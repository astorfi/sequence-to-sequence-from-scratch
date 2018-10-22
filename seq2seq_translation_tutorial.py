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
from data_loader import Dataset
from _utils import transformer
import argparse


# Predefined tokens
SOS_token = 1   # SOS_token: start of sentence
EOS_token = 2   # EOS_token: end of sentence

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")

# Parser
parser = argparse.ArgumentParser(description='Creating Classifier')

######################
# Optimization Flags #
######################

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--epochs_per_lr_drop', default=450, type=float,
                    help='number of epochs for which the learning rate drops')

##################
# Training Flags #
##################
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_epoch', default=600, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--save_folder', default=os.path.expanduser('~/weights'), help='Location to save checkpoint models')
parser.add_argument('--epochs_per_save', default=10, type=int,
                    help='number of epochs for which the model will be saved')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')

###############
# Model Flags #
###############

parser.add_argument('--auto_encoder', default=True, type=str2bool, help='Use auto-encoder model')
parser.add_argument('--MAX_LENGTH', default=10, type=int, help='Maximum length of sentence')
parser.add_argument('--bidirectional', default=False, type=str2bool, help='bidirectional LSRM')

# Add all arguments to parser
args = parser.parse_args()

##############
# Cuda Flags #
##############
if args.cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

###############################
# Creating the dataset object #
###############################
# Create training data object
trainset = Dataset(phase='train', max_input_length=10)

# Extract the languages' attributes
input_lang, output_lang = trainset.langs()

# The trainloader for parallel processing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
# iterate through training
dataiter = iter(trainloader)

# Create testing data object
testset = Dataset(phase='test', max_input_length=10)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=True, num_workers=1, pin_memory=False, drop_last=True)

#####################
# Encoder / Decoder #
#####################

class EncoderRNN(nn.Module):
    """
    The encoder generates a single output vector that embodies the input seqence meaning.
    The general procedure is as follows:
        1. In each step, a word will be fed to a network and it generates
         an output and a hidden state.
        2. For the next step, the hidden step and the next word will
         be fed to the same network (W) for updating the weights.
        3. In the end, the last output will be the representative of the input sentence (called the "context vector").
    """
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1, bidirectional=1):
        """
        * For nn.LSTM, same input_size & hidden_size is chosen.
        :param input_size: The size of the input vocabulary
        :param hidden_size: The hidden size of the RNN.
        :param batch_size: The batch0size for mini-batch optimization.
        :param num_layers: Number of RNN layers. Default: 1
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.num_layers)


    def forward(self, input, hidden):
        # Make the data in the correct format as the RNN input.
        embedded = self.embedding(input).view(1, 1, -1)
        rnn_input = embedded
        # The following descriptions of shapes and tensors are extracted from the official Pytorch documentation:
        # output-shape: (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM
        # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state
        # c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state
        output, (h_n, c_n) = self.lstm(rnn_input, hidden)
        return output, (h_n, c_n)

    def initHidden(self):

        if self.bidirectional:
            encoder_hidden = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                                      torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
            encoder_hidden = {"forward": encoder_hidden, "backward": encoder_hidden}
            return encoder_hidden
        else:
            encoder_hidden = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                              torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
            return encoder_hidden

class DecoderRNN(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    Decoding is as follows:
    1. At each step, an input token and a hidden state is fed to the decoder.
        * The initial input token is the <SOS>.
        * The first hidden state is the context vector generated by the encoder (the encoder's
    last hidden state).
    2. The first output, shout be the first sentence of the output and so on.
    3. The input token sequence ends with <EOS> token.
    """
    def __init__(self, hidden_size, output_size, batch_size, num_layers=1, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.lstm = nn.LSTM(input_size= hidden_size, hidden_size=hidden_size, num_layers=(int(self.bidirectional) + 1) * self.num_layers)
        self.lstm = nn.LSTM(input_size= hidden_size, hidden_size=hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, (h_n, c_n) = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, (h_n, c_n)

    def initHidden(self):
        """
        The spesific type of the hidden layer for the RNN type that is used (LSTM).
        :return: All zero hidden state.
        """
        num_directions = int(self.bidirectional) + 1
        return [torch.zeros(self.num_layers * num_directions, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers * num_directions, 1, self.hidden_size, device=device)]


######################
# Training the Model #
######################
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=args.MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    encoder_hiddens_last = []
    loss = 0

    for step_idx in range(args.batch_size):
        # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
        # the new input sequence as a continuation of the previous sequence
        encoder_hidden = encoder.initHidden()
        input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]
        input_length = input_tensor_step.size(0)

        if args.bidirectional:
            encoder_outputs = torch.zeros(args.batch_size, max_length, 2 * encoder.hidden_size, device=device)
            encoder_hidden_forward = encoder_hidden['forward']
            encoder_hidden_backward = encoder_hidden['backward']
            for ei in range(input_length):
                encoder_output, encoder_hidden_forward = encoder(
                    input_tensor_step[ei], encoder_hidden_forward)
                encoder_outputs[step_idx, ei, 0:encoder.hidden_size] = encoder_output[0, 0]
            for ei in range(input_length):
                encoder_output, encoder_hidden_backward = encoder(
                    input_tensor_step[input_length - 1 - ei], encoder_hidden_backward)
                encoder_outputs[step_idx, ei, encoder.hidden_size:] = encoder_output[0, 0]
            encoder_cn = torch.cat((encoder_hidden_forward[0], encoder_hidden_backward[0]), 0)
            encoder_hn = torch.cat((encoder_hidden_forward[1], encoder_hidden_backward[1]), 0)
            encoder_hidden = [encoder_cn, encoder_hn]


        else:
            encoder_outputs = torch.zeros(args.batch_size, max_length, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor_step[ei], encoder_hidden)
                encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]

        encoder_hiddens_last.append([encoder_output, encoder_output])

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hiddens = encoder_hiddens_last

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:

        for step_idx in range(args.batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                decoder_input = target_tensor_step[di]  # Teacher forcing

        loss = loss / args.batch_size

    else:
        for step_idx in range(args.batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                #     decoder_input, decoder_hidden, encoder_outputs)
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                if decoder_input.item() == EOS_token:
                    break
        loss = loss / args.batch_size

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.

def reformat_tensor_(tensor):
    tensor = tensor.transpose(0, 2, 1)
    tensor = tensor.squeeze()
    return tensor[tensor != -1].view(-1, 1)

def reformat_tensor_mask(tensor):
    tensor = tensor.squeeze(dim=1)
    tensor = tensor.transpose(1,0)
    mask = tensor != 0
    return tensor, mask



def trainIters(encoder, decoder, print_every=1000, plot_every=100, learning_rate=0.1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    n_iters_per_epoch = int(len(trainset) / args.batch_size)
    for i in range(num_epochs):

        for iteration, data in enumerate(trainloader, 1):

            # Get a batch
            training_pair = data

            # Input
            input_tensor = training_pair['sentence'][:,:,0,:]
            input_tensor, mask_input = reformat_tensor_mask(input_tensor)

            # Target
            target_tensor = training_pair['sentence'][:,:,1,:]
            target_tensor, mask_target = reformat_tensor_mask(target_tensor)

            if device == torch.device("cuda"):
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            loss = train(input_tensor, target_tensor, mask_input, mask_target, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iteration % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iteration / n_iters_per_epoch),
                                             iteration, iteration / n_iters_per_epoch * 100, print_loss_avg))

            if iteration % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            # break
        print('####### Finished epoch %d of %d ########' % (i+1, num_epochs))

    showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, input_tensor, max_length=args.MAX_LENGTH):
    with torch.no_grad():

        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        # return decoded_words, decoder_attentions[:di + 1]
        return decoded_words


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = testset[i]['sentence']
        input_tensor, mask_input = reformat_tensor_mask(pair[:,0,:].view(1,1,-1))
        input_tensor = input_tensor[input_tensor != 0]
        output_tensor, mask_output = reformat_tensor_mask(pair[:,1,:].view(1,1,-1))
        output_tensor = output_tensor[output_tensor != 0]
        if device == torch.device("cuda"):
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        input_sentence = ' '.join(transformer.SentenceFromTensor_(input_lang, input_tensor))
        output_sentence = ' '.join(transformer.SentenceFromTensor_(output_lang, output_tensor))
        print('Input: ', input_sentence)
        print('Output: ', output_sentence)
        output_words = evaluate(encoder, decoder, input_tensor)
        output_sentence = ' '.join(output_words)
        print('Predicted Output: ', output_sentence)
        print('')


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, args.batch_size, num_layers=3, bidirectional=args.bidirectional).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words, args.batch_size, num_layers=3, bidirectional=args.bidirectional).to(device)

trainIters(encoder1, decoder1, print_every=10)

######################################################################
#

evaluateRandomly(encoder1, decoder1)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())
#
#
# ######################################################################
# # For a better viewing experience we will do the extra work of adding axes
# # and labels:
# #
#
# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
#
#
# evaluateAndShowAttention("elle a cinq ans de moins que moi .")
#
# evaluateAndShowAttention("elle est trop petit .")
#
# evaluateAndShowAttention("je ne crains pas de mourir .")
#
# evaluateAndShowAttention("c est un jeune directeur plein de talent .")


######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#
