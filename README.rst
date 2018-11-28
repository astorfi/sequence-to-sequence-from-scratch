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
