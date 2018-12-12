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

The goal here is to create a **sequence-to-sequence mapping** model which is going to be built on an
Encoder-Decoder network. The model encode the information into a specific representation. This representation
later on will be mapped as a target output sequence. This transition makes the model understand the interoperibility
between two sequences. In another word, the meaningful connection between the two sequence will be created. Two important
sequence to sequence modeling examples are ``Machine Transtional`` and ``Autoencoders``. Here, we can do both just by
chaning the ``input-output`` language sequences.

------------------
Word Embedding
------------------

At the very first step, we should know what are the ``input-output sequences`` and how we should ``represent the data``
for the model to understand it. Clearly, it should be a sequence of words in the input and the equivalent
sequence in the output. In case of having an autoencoder, both input and output sentences
are the same.

A learned representation for context elements is called ``word embedding`` in which the words with similar meaning, ideally,
become highly correlated in the representation space as well. One of the main incentives behind word embedding representations
is the high generalization power as opposed to sparse higher dimensional representation [goldberg2017neural]_. Unlike the traditional
bag-of-word representation in which different words have quite different representation regardless of their usage,
in learning the distributed representation, the usage of words in the context is of great importance which lead to
similar representation for correlated words in meaning. The are different approaches for creating word embedding. Please
refer to the great Pytorch tutorial titled [`WORD EMBEDDINGS: ENCODING LEXICAL SEMANTICS <https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html>`_]
for more details.

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
and returned). The created feature vector will represents the initial hidden states of the decoder. The
architecture of a bi-lstm is as below:

.. figure:: _img/bilstm.png
   :scale: 50
   :alt: map to buried treasure

**NOTE:** As can be observered in the figure *colors*, two ``independent`` different set of
 weights ``MUST`` be considered for the forward and backward passes, Otherwise, the network will
 assume the backward pass follows the forward pass!!

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
[`Understanding LSTM Netwroks <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_].
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

As it can be seen in the above code, for the *Bidirectional LSTM*, we have **separate and independent**
states for ``forwards`` and ``backward`` directions.


-----------------------------
Decoder
-----------------------------

For the decoder, the final encoder hidden state (or the concatenation if we have a bi-lstm as the encoder)
of the encoder will be called ``context vector``. This context vector, generated by the encoder, will
be used as the initial hidden state of the decoder. Decoding is as follows:

    1. At each step, an input token and a hidden state is fed to the decoder.

        * The initial input token is the ``<SOS>``.
        * The first hidden state is the context vector generated by the encoder (the encoder's last hidden state).

    2. The first output, should be the first word of the output sequence and so on.
    3. The output token generation ends with ``<EOS>`` being generated or the predefined max_length of the output sentence.

After the first decoder step, for the following steps, the input is going to be the previous word prediction of the RNN.
So the output generation will be upon the network sequence prediction. In case of using ``teacher_forcing``, the input is going to be the actual
targeted output word. It provides better guidance for the training but it is inconsistent with the evaluation stage as
targeted outputs do not exists! In order to handle the issue with this approach, new approaches have been proposed [lamb2016professor]_.

The decoder, will generally be initialized as below:

.. code-block:: python

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

-------------------------------
Encoder-Decoder Bridge
-------------------------------

The context vector, generated by the encoder, will be used as the initial hidden state of the decoder.
In case that their *dimension is not matched*, a ``linear layer`` should be employed to transformed the context vector
to a suitable input (shape-wise) for the decoder cell state (including the memory(Cn) and hidden(hn) states).
The shape mismatch is True in the following conditions:

    1. The hidden sizes of encoder and decoder are the same BUT we have a bidirectional LSTM as the Encoder.
    2. The hidden sizes of encoder and decoder are NOT same.
    3. ETC?


The linear layer will be defined as below:

.. code-block:: python

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


============
Training
============

***************
References
***************

https://medium.com/datadriveninvestor/neural-translation-model-95277838d17d

.. [goldberg2017neural] Goldberg, Yoav. "Neural network methods for natural language processing." Synthesis Lectures on Human Language Technologies 10.1 (2017): 1-309.
.. [lamb2016professor] Lamb, A.M., GOYAL, A.G.A.P., Zhang, Y., Zhang, S., Courville, A.C. and Bengio, Y., 2016. Professor forcing: A new algorithm for training recurrent networks. In Advances In Neural Information Processing Systems (pp. 4601-4609).
