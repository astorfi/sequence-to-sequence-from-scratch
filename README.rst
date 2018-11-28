##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4

***************
Documentation
***************

============
Dataset
============

============
Model
============

------------------------------------------------------------
Encoder
------------------------------------------------------------

The encoder generates a single output vector that embodies the input sequence meaning. The general procedure is as follows:

    1. In each step, a word will be fed to a network and it generates an output and a hidden state.
    2. For the next step, the hidden step and the next word will be fed to the same network (W) for updating the weights.
    3. In the end, the last output will be the representative of the input sentence (called the "context vector").

The ``EncoderRNN`` attribute is dedicated to the encoder structure. The Encoder in our code,
can be a ``unidirectional/bidirectional LSTM``. A *Bidirectional* LSTM consists of *two
independent LSTMs*, one take the input sequence in normal time order and the other one
will be fed with the input sequence in the reverse time order. The outputs of the two
will usually be concatenated at each time step (usually the *last hidden states* will be concatenated
and returned). The created feature vector will represents the initial hidden states of the decoder.

The encoder, will generally be initialized as below:

.. code-block:: python

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
     self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)


**NOTE:** We ``do NOT`` generate the whole LSTM/Bi-LSTM architecture using Pytorch. Instead, we just use
the LSTM cells to represent **what exactly is going on in the encoding/decoding** phases!

The initialization of the LSTM is a little bit different compared to the LSTM
[`Understanding LSTM Netwroks <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_ ].
Both cell state and hidden states must be initialized as belows:

.. code-block:: python

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
