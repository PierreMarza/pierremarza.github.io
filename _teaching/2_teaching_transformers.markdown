---
layout: distill
title: Introduction to Deep Learning
description: Transformers and Attention
date: 2023-09-25

authors:
  - name: Pierre Marza
    url: "https://pierremarza.github.io/"
    affiliations:
      name: INSA, Lyon

bibliography: 2018-12-22-distill.bib

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

# 1. Lecture
The lecture slides are available [here](https://pierremarza.github.io/teaching/epita_ing3_deep_learning_transformers.pdf).

# 2. Practical
## Context
The goals of this session are to practice with **implementing a Transformer model from scratch**, **understanding the involved computations**, and more generally, to **build a full Deep Learning pipeline in PyTorch** to train a model on a given dataset.

This practical session is mostly about **implementing a Transformer encoder from scratch**, more precisely reproducing the encoder in the [**Attention is all you need**](https://arxiv.org/pdf/1706.03762.pdf) paper. The downstream task will be the same as in the previous practical session: **predicting the country of origin of an input name**.

## Installation
We will be coding with **Python3** and will use the **Pytorch** library.

To install Pytorch on your local machine, follow this [link](https://pytorch.org/get-started/locally/)

## Transformers
Transformes have recently become a go-to solution when dealing with sequential data. You can refer to your [lecture](https://pierremarza.github.io/teaching/epita_ing3_deep_learning_transformers.pdf) for more information about Transformers, as well as many great online resources.

## Data
We will use the same data as in the previous practical session. You can re-use your custom *Dataset* class and dataloaders.

## Implementing a Transformer model
You will have 3 different classes:
* *TransformerEncoder*: your full Transformer encoder composed of a positional encoding layer and a sequence of self-attention layers.
* *PositionalEncoding* (already implemented for you): layer predicting and adding positional encodings to the inputs.
* *TransformerEncoderLayer*: a self-attention block.

The structure of the classes to implement is given below:

```python
import math
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, emsize, d_hid, nlayers, dropout):
        """
        Arguments:
            emsize: int, size of the token embeddings.
            d_hid: int, size of hidden embeddings in the self-attention forward pass.
            nlayers: int, number of self-attention layers.
            dropout: float, dropout probability.
        """
        super().__init__()
        # First projection layer
        self.proj_layer = nn.Linear(emsize, d_hid)

        # TODO: Init what you will need in different methods.
        pass

    def concat_cls(self, x):
        # TODO: concatenate a learnt 'CLS' token to the
        # input sequence x and return the new sequence.
        pass

    def add_positional_encoding(self, x):
        # TODO: return the sequence token x after adding 
        # positional encoding to them.
        pass

    def forward(self, x):
        x = self.proj_layer(x)

        # TODO: Implement the forward pass of the Transformer.
        pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Init what you will need in different methods.
        pass

    def predicting_qkv(self, x):
        # TODO: return the queries, keys and values from 
        # input tokens x.
        pass

    def self_attention(self, q, k, v, d_k, mask):
        # TODO: return the new representation of all tokens 
        # given keys, queries, values and key dimension as 
        # inputs. You can use a mask to discard masked
        # tokens.
        pass

    def norm(self, x):
        # TODO: return the normalized input (LayerNorm).
        pass

    def feed_forward(self, x):
        # TODO: return the output of the Feed forward 
        # layer.
        pass

    def forward(self, x):
        # TODO: Implement the forward pass of the encoder 
        # layer.
        pass

```

### Concatenating a learned 'CLS' token
Implement the *concat_cls* method that takes the sequence of tokens as input and simply concatenates the 'CLS' token. In order to predict the embedding of the 'CLS' token, use a [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layer.

### Computing positional encoding
Implement the *add_positional_encoding* method that computes positional encodings and adds them to all tokens in the sequence.

### Predicting queries, keys and values from input tokens
Implement the *predicting_qkv* method that predicts queries, keys and values for all tokens in the sequence.

### Self-attention
Implement the *self_attention* method that performs the scaled dot-product attention and outputs the new embedding for all tokens in the sequence. You can use a mask token here to discrad masked tokens in the sequence.

### Norm
Implement the *norm* layer that computes a LayerNorm operation on the input. You can use Pytorch [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

### Write the forward pass
Implement the full forward pass of the attention layer (Positional Encoding, Self-Attention, Add & Norm, Feed forward, Add & Norm). Don't forget the residual connections ("Add" in "Add & Norm")!

## Loss function and optimizer
The next step is to define a [**loss function**](https://pytorch.org/docs/stable/nn.html#loss-functions) that is suited to the problem. Then you have to choose an [**optimizer**](https://pytorch.org/docs/stable/optim.html). You are encouraged to try different ones to compare them. You can also study the impact of different hyperparameters of the optimizer (learning rate, momentum, etc.)

## Training loop
It is now to time to write the code for **training and validating your model**. You must iterate through your training data using your dataloader, and compute forward and backward passes on given data batches.
Don't forget to log your training as well as validation losses (the latter is mainly used to tune hyperparameters).

## Visualizing your training with Tensorboard
A useful tool to visualize your training is [**Tensorboard**](https://www.tensorflow.org/tensorboard/). You can also have a look at solutions such as [**Weights & Biases**](https://wandb.ai/site), but we will focus on the simpler Tensorboard for now.
You can easily use Tensorboard with Pytorch by looking at [**torch.utils.tensorboard**](https://pytorch.org/docs/stable/tensorboard.html)

## Saving and loading a Pytorch model
Once training is completed, it can be useful to save the weights of your neural network to use it later. The following [tutorial](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) explains how you can do this. Now, try to save and then load your trained model.

## Testing your model
You must now **evaluate the performance of your trained model** on the **test set**. To this end, you have to iterate through test samples, and perform forward passes on given data batches. You might want to compute the **test loss**, but also any **accuracy-related metrics** you are interested in. You could also **visualize some test samples** along with the **output distribution of your model**.

## Comparing your custom implementation with Pytorch transformer layers
In this final part, replace your custom implementation of a Transformer layer with [**nn.TransformerEncoderLayer**](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) to see if there are differences in final model performance.