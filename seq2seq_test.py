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
from _utils.transformer import *
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
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs for training for training')
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
parser.add_argument('--hidden_size_decoder', default=256, type=int, help='Decoder Hidden Size')
parser.add_argument('--num_layer_decoder', default=1, type=int, help='Number of LSTM layers for decoder')
parser.add_argument('--hidden_size_encoder', default=256, type=int, help='Eecoder Hidden Size')
parser.add_argument('--num_layer_encoder', default=1, type=int, help='Number of LSTM layers for encoder')
parser.add_argument('--teacher_forcing', default=False, type=str2bool, help='If using the teacher frocing in decoder')

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
trainset = Dataset(phase='train', max_input_length=10, auto_encoder=args.auto_encoder)

# Extract the languages' attributes
input_lang, output_lang = trainset.langs()

# The trainloader for parallel processing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
# iterate through training
dataiter = iter(trainloader)

# Create testing data object
testset = Dataset(phase='test', max_input_length=10, auto_encoder=args.auto_encoder)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=True, num_workers=1, pin_memory=False, drop_last=True)

#####################
# Encoder / Decoder #
#####################

class EncoderRNN(nn.Module):
    """
    The encoder generates a single output vector that embodies the input sequence meaning.
    The general procedure is as follows:
        1. In each step, a word will be fed to a network and it generates
         an output and a hidden state.
        2. For the next step, the hidden step and the next word will
         be fed to the same network (W) for updating the weights.
        3. In the end, the last output will be the representative of the input sentence (called the "context vector").
    """
    def __init__(self, hidden_size, input_size, batch_size, num_layers=1, bidirectional=False):
        """
        * For nn.LSTM, same input_size & hidden_size is chosen.
        :param input_size: The size of the input vocabulary
        :param hidden_size: The hidden size of the RNN.
        :param batch_size: The batch_size for mini-batch optimization.
        :param num_layers: Number of RNN layers. Default: 1
        :param bidirectional: If the encoder is a bi-directional LSTM. Default: False
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        # The input should be transformed to a vector that can be fed to the network.
        self.embedding = nn.Embedding(input_size, embedding_dim=hidden_size)

        # The LSTM layer for the input
        if args.bidirectional:
            self.lstm_forward = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
            self.lstm_backward = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)


    def forward(self, input, hidden):

        if args.bidirectional:
            input_forward, input_backward = input
            hidden_forward, hidden_backward = hidden
            input_forward = self.embedding(input_forward).view(1, 1, -1)
            input_backward = self.embedding(input_backward).view(1, 1, -1)

            out_forward, (h_n_forward, c_n_forward) = self.lstm_forward(input_forward, hidden_forward)
            out_backward, (h_n_backward, c_n_backward) = self.lstm_backward(input_backward, hidden_backward)

            forward_state = (h_n_forward, c_n_forward)
            backward_state = (h_n_backward, c_n_backward)
            output_state = (forward_state, backward_state)

            return output_state
        else:
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
            encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                                      torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
            encoder_state = {"forward": encoder_state, "backward": encoder_state}
            return encoder_state
        else:
            encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                              torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]
            return encoder_state

class DecoderRNN(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    Decoding is as follows:
    1. At each step, an input token and a hidden state is fed to the decoder.
        * The initial input token is the <SOS>.
        * The first hidden state is the context vector generated by the encoder (the encoder's
    last hidden state).
    2. The first output, shout be the first sentence of the output and so on.
    3. The output token generation ends with <EOS> being generated or the predefined max_length of the output sentence.
    """
    def __init__(self, hidden_size, output_size, batch_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
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
        return [torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device)]

class Linear(nn.Module):
    """
    This context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
    In case that their dimension is not matched, a linear layer should be used to transformed the context vector
    to a suitable input (shape-wise) for the decoder cell state (including the memory(Cn) and hidden(hn) states).
    The shape mismatch is True in the following conditions:
    1. The hidden sizes of encoder and decoder are the same BUT we have a bidirectional LSTM as the Encoder.
    2. The hidden sizes of encoder and decoder are NOT same.
    3. ETC?
    """

    def __init__(self, bidirectional, hidden_size_encoder, hidden_size_decoder):
        super(Linear, self).__init__()
        self.bidirectional = bidirectional
        num_directions = int(bidirectional) + 1
        self.linear_connection_op = nn.Linear(num_directions * hidden_size_encoder, hidden_size_decoder)
        self.connection_possibility_status = num_directions * hidden_size_encoder == hidden_size_decoder

    def forward(self, input):

        if self.connection_possibility_status:
            return input
        else:
            return self.linear_connection_op(input)


######################
# Training the Model #
######################

def train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer, criterion, max_length=args.MAX_LENGTH):
    # To train, each element of the input sentence will be fed to the encoder.
    # At the decoding phase``<SOS>`` will be fed as the first input to the decoder
    # and the last hidden (state,cell) of the encoder will play the role of the first hidden (cell,state) of the decoder.

    # optimizer steps
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    bridge_optimizer.zero_grad()

    # Define a list for the last hidden layer
    encoder_hiddens_last = []
    loss = 0

    #################
    #### DECODER ####
    #################
    for step_idx in range(args.batch_size):
        # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
        # the new input sequence as a continuation of the previous sequence.
        encoder_hidden = encoder.initHidden()
        input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]
        input_length = input_tensor_step.size(0)

        # Switch to bidirectional mode
        if args.bidirectional:
            encoder_outputs = torch.zeros(args.batch_size, max_length, 2 * encoder.hidden_size, device=device)
            encoder_hidden_forward = encoder_hidden['forward']
            encoder_hidden_backward = encoder_hidden['backward']
            for ei in range(input_length):
                (encoder_hidden_forward, encoder_hidden_backward) = encoder(
                    (input_tensor_step[ei],input_tensor_step[input_length - 1 - ei]), (encoder_hidden_forward,encoder_hidden_backward))

            # Extract the hidden and cell states
            hn_forward, cn_forward = encoder_hidden_forward
            hn_backward, cn_backward = encoder_hidden_backward

            # Concatenate the hidden and cell states for forward and backward paths.
            encoder_hn = torch.cat((hn_forward, hn_backward), 2)
            encoder_cn = torch.cat((cn_forward, cn_backward), 2)


            # Only return the hidden and cell states for the last layer and pass it to the decoder
            encoder_hn_last_layer = encoder_hn[-1].view(1, 1, -1)
            encoder_cn_last_layer = encoder_cn[-1].view(1,1,-1)

            # The list of states
            encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]


        else:
            encoder_outputs = torch.zeros(args.batch_size, max_length, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor_step[ei], encoder_hidden)
                encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]

            # only return the hidden and cell states for the last layer and pass it to the decoder
            hn, cn = encoder_hidden
            encoder_hn_last_layer = hn[-1].view(1,1,-1)
            encoder_cn_last_layer = cn[-1].view(1,1,-1)
            encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]

        # A linear layer to establish the connection between the encoder/decoder layers.
        encoder_hidden = [bridge(item) for item in encoder_hidden]
        encoder_hiddens_last.append(encoder_hidden)

    #################
    #### DECODER ####
    #################
    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hiddens = encoder_hiddens_last

    # teacher_forcing uses the real target outputs as the next input
    # rather than using the decoder's prediction.
    if args.teacher_forcing:

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




def trainIters(encoder, decoder, bridge, print_every=1000, plot_every=100, learning_rate=0.1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    bridge_optimizer = optim.SGD(bridge.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    n_iters_per_epoch = int(len(trainset) / args.batch_size)
    for i in range(args.num_epochs):

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
                         decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer, criterion)
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


        print('####### Finished epoch %d of %d ########' % (i+1, args.num_epochs))


##############
# Evaluation #
##############
#
# In evaluation, we simply feed the sequence and observe the output.
# The generation will be over once the "EOS" has been generated.

def evaluate(encoder, decoder, bridge, input_tensor, max_length=args.MAX_LENGTH):

    # Required for tensor matching.
    # Remove to see the results for educational purposes.
    with torch.no_grad():

        # Initialize the encoder hidden.
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        if args.bidirectional:
            encoder_outputs = torch.zeros(args.batch_size, max_length, 2 * encoder.hidden_size, device=device)
            encoder_hidden_forward = encoder_hidden['forward']
            encoder_hidden_backward = encoder_hidden['backward']

            for ei in range(input_length):
                (encoder_hidden_forward, encoder_hidden_backward) = encoder(
                    (input_tensor[ei],input_tensor[input_length - 1 - ei]), (encoder_hidden_forward,encoder_hidden_backward))

            # Extract the hidden and cell states
            hn_forward, cn_forward = encoder_hidden_forward
            hn_backward, cn_backward = encoder_hidden_backward

            # Concatenate the hidden and cell states for forward and backward paths.
            encoder_hn = torch.cat((hn_forward, hn_backward), 2)
            encoder_cn = torch.cat((cn_forward, cn_backward), 2)


            # Only return the hidden and cell states for the last layer and pass it to the decoder
            encoder_hn_last_layer = encoder_hn[-1].view(1, 1, -1)
            encoder_cn_last_layer = encoder_cn[-1].view(1,1,-1)

            # The list of states
            encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        else:
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)

            # only return the hidden and cell states for the last layer and pass it to the decoder
            hn, cn = encoder_hidden
            encoder_hn_last_layer = hn[-1].view(1,1,-1)
            encoder_cn_last_layer = cn[-1].view(1,1,-1)
            encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        decoder_input = torch.tensor([SOS_token], device=device)  # SOS
        encoder_hidden_last = [bridge(item) for item in encoder_hidden_last]
        decoder_hidden = encoder_hidden_last

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
def evaluateRandomly(encoder, decoder, bridge, n=10):
    for i in range(n):
        pair = testset[i]['sentence']
        input_tensor, mask_input = reformat_tensor_mask(pair[:,0,:].view(1,1,-1))
        input_tensor = input_tensor[input_tensor != 0]
        output_tensor, mask_output = reformat_tensor_mask(pair[:,1,:].view(1,1,-1))
        output_tensor = output_tensor[output_tensor != 0]
        if device == torch.device("cuda"):
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        input_sentence = ' '.join(SentenceFromTensor_(input_lang, input_tensor))
        output_sentence = ' '.join(SentenceFromTensor_(output_lang, output_tensor))
        print('Input: ', input_sentence)
        print('Output: ', output_sentence)
        output_words = evaluate(encoder, decoder, bridge, input_tensor)
        output_sentence = ' '.join(output_words)
        print('Predicted Output: ', output_sentence)
        print('')


######################################################################
# Training and Evaluating
# =======================

encoder1 = EncoderRNN(args.hidden_size_encoder, input_lang.n_words, args.batch_size, num_layers=args.num_layer_encoder, bidirectional=args.bidirectional).to(device)
bridge = Linear(args.bidirectional, args.hidden_size_encoder, args.hidden_size_decoder).to(device)
decoder1 = DecoderRNN(args.hidden_size_decoder, output_lang.n_words, args.batch_size, num_layers=args.num_layer_decoder).to(device)

trainIters(encoder1, decoder1, bridge, print_every=10)

######################################################################
evaluateRandomly(encoder1, decoder1, bridge)
