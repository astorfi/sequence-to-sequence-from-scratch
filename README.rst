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

The ``EncoderRNN`` attribute is dedicated to the encoder structure. The
Encoder in our code, can be a ``unidirectional/bidirectional LSTM``. A Bidirectional LSTM
consists of two independent LSTMs, one take the input sequence in normal time order
and the other one will be fed with the input sequence in the reverse time order.
The outputs of the two will usually be concatenated at each time step.
