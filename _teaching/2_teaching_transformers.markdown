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

{::options parse_block_html="true" /}
<details><summary markdown="span">**A solution**</summary>
```python
import torch
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, emsize, d_hid, nlayers, dropout):
        super().__init__()
        self.emsize = emsize
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout

        # First projection layer
        self.proj_layer = nn.Linear(emsize, d_hid)

        # concat_cls
        self.cls_embedding_layer = nn.Embedding(num_embeddings=1, embedding_dim=d_hid)

        # add_positional_encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_hid, max_len=20, dropout=dropout
        )

        # encoder layers
        self.encoder = TransformerEncoderLayer(d_hid=d_hid, dq=64, dk=64, dv=64)

        # Classif
        self.classif = nn.Linear(d_hid, 18)

    def concat_cls(self, x):
        bs, device = x.shape[0], x.device
        cls_token = self.cls_embedding_layer(torch.zeros((bs, 1)).long().to(device))
        x = torch.cat([cls_token, x], dim=1)
        return x

    def add_positional_encoding(self, x):
        return self.pos_encoding(x)

    def forward(self, x, mask):
        x = self.proj_layer(x)
        x = self.concat_cls(x)
        x = self.add_positional_encoding(x)
        x = self.encoder(x, mask)
        return self.classif(x[:, 0])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_hid, dq, dk, dv):
        super().__init__()

        # Attention
        self.linear_q = nn.Linear(d_hid, dq)
        self.linear_k = nn.Linear(d_hid, dk)
        self.linear_v = nn.Linear(d_hid, dv)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_hid)

        # Feed forward
        self.feed_forward_layer = nn.Sequential(
            nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid)
        )

    def predicting_qkv(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return q, k, v

    def self_attention(self, q, k, v, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v)

    def norm(self, x):
        return self.layer_norm(x)

    def feed_forward(self, x):
        return self.feed_forward_layer(x)

    def forward(self, x, mask):
        q, k, v = self.predicting_qkv(x)
        out_attention = self.self_attention(q, k, v, mask)
        out_attention = self.norm(x + out_attention)

        out_feed_forward = self.feed_forward(out_attention)
        out_feed_forward = self.norm(out_attention + out_feed_forward)

        return out_feed_forward

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samples, max_line_len):
        self.samples = samples
        self.max_line_len = max_line_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_dict = self.samples[index]

        name = lineToTensor(sample_dict["name"])

        #####################################################################################
        ## Same dataset as in previous practical, with simply a different mask computation
        mask = torch.zeros((self.max_line_len + 1, self.max_line_len + 1))
        mask[: name.shape[0] + 1, : name.shape[0] + 1] = 1
        #####################################################################################

        name = torch.cat(
            [
                name,
                torch.zeros(
                    (self.max_line_len - name.shape[0], name.shape[1], name.shape[2])
                ),
            ],
            dim=0,
        )

        label = sample_dict["label"]
        label = torch.Tensor([label])
        return {"name": name, "label": label, "mask": mask}

train_dataset = CustomDataset(train_samples, max_line_len)
val_dataset = CustomDataset(val_samples, max_line_len)
test_dataset = CustomDataset(test_samples, max_line_len)

batch_size = 16
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

model = TransformerEncoder(emsize=57, d_hid=64, nlayers=2, dropout=0.1)
```
</details>
<br/>
{::options parse_block_html="false" /} 

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

{::options parse_block_html="true" /}
<details><summary markdown="span">**A solution**</summary>
```python
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```
</details>
<br/>
{::options parse_block_html="false" /} 

## Training loop
It is now to time to write the code for **training and validating your model**. You must iterate through your training data using your dataloader, and compute forward and backward passes on given data batches.
Don't forget to log your training as well as validation losses (the latter is mainly used to tune hyperparameters).

{::options parse_block_html="true" /}
<details><summary markdown="span">**A solution**</summary>
```python
def train_val(run_type, criterion, dataloader, model, optimizer):
    tot_loss = 0.0
    tot_acc = []
    for mb_idx, batch in tqdm(enumerate(dataloader)):
        name = batch["name"].squeeze(2)
        label = batch["label"].squeeze(1).long()
        mask = batch["mask"]

        if run_type == "train":
            # zero the parameter gradients
            optimizer.zero_grad()

        # Forward pass
        if run_type == "train":
            out = model(name, mask=mask)
        elif run_type == "val":
            with torch.no_grad():
                out = model(name, mask=mask)

        # Compute loss
        loss = criterion(out, label)

        if run_type == "train":
            # Compute gradients
            loss.backward()

            # Backward pass - model update
            optimizer.step()

        # Logging
        tot_loss += loss.item()
        acc = (out.argmax(dim=1) == label).tolist()
        tot_acc.extend(acc)
    return tot_loss, tot_acc, criterion, dataloader, model, optimizer


epochs = 10
for epoch in range(epochs):
    # Training
    epoch_loss, epoch_acc, criterion, train_dataloader, model, optimizer = train_val(
        "train", criterion, train_dataloader, model, optimizer
    )
    print(
        f"Epoch {epoch}: {epoch_loss/len(train_dataloader)}, {np.array(epoch_acc).mean()}"
    )

    # Validation
    val_loss, val_acc, criterion, val_dataloader, model, optimizer = train_val(
        "val", criterion, val_dataloader, model, optimizer
    )
    print(f"Val: {val_loss/len(val_dataloader)}, {np.array(val_acc).mean()}")
```
</details>
<br/>
{::options parse_block_html="false" /} 

## Visualizing your training with Tensorboard
A useful tool to visualize your training is [**Tensorboard**](https://www.tensorflow.org/tensorboard/). You can also have a look at solutions such as [**Weights & Biases**](https://wandb.ai/site), but we will focus on the simpler Tensorboard for now.
You can easily use Tensorboard with Pytorch by looking at [**torch.utils.tensorboard**](https://pytorch.org/docs/stable/tensorboard.html)

## Saving and loading a Pytorch model
Once training is completed, it can be useful to save the weights of your neural network to use it later. The following [tutorial](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) explains how you can do this. Now, try to save and then load your trained model.

## Testing your model
You must now **evaluate the performance of your trained model** on the **test set**. To this end, you have to iterate through test samples, and perform forward passes on given data batches. You might want to compute the **test loss**, but also any **accuracy-related metrics** you are interested in. You could also **visualize some test samples** along with the **output distribution of your model**.

## Comparing your custom implementation with Pytorch transformer layers
In this final part, replace your custom implementation of a Transformer layer with [**nn.TransformerEncoderLayer**](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) to see if there are differences in final model performance.