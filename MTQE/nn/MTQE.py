#!/usr/bin/env python
# coding=utf-8

import math
import torch
from torch import nn
from transformers import XLMRobertaModel

class MHA(nn.Module):
    """Class implementing Multi Head Attention with a couple of tweaks.
    Largely inspired by the implementation of JoeyNMT (https://github.com/joeynmt/joeynmt).
    """

    def __init__(self, num_heads, size, dropout):
        """MHA constructor.
    
        Positional arguments:
        - num_heads: the number of heads 
        - size: model size (embedding dimensions)
        - dropout: dropout to be applied for attention
        """
        super(MHA, self).__init__()
        assert size % num_heads == 0
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v, q, mask):
        """Forward function for MHA.
        Usual attention values normalization is softmax (sum(scores) = 1) as the commented line below
        attention = self.softmax(scores)
        But we replaced it by sigmoid normalizaton (each score in [0;1])
        We thus use the logistic function as presented in
        - Sheng Syun Shen and Hung Yi Lee. 2016. 
            Neural attention models for sequence classification: Analysis and application to key term extraction 
            and dialogue act detection
        and in 
        - Marek Rei and Anders Søgaard. 2018. Zero-shot Sequence Labeling: Transferring Knowledge from Sentences to Tokens

        Positional arguments:
        - k: keys
        - v: values
        - q: queries
        - mask: mask for padding
        """
        batch_size = k.shape[0]
        num_heads = self.num_heads

        # project the queries, keys and values
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v to [batch_size, seq_len, num_heads, model_size / num_heads]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # compute the scores (shape is [batch_size, num_heads, seq_len, seq_len])
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask after unsqueezing it to shape [batch_size, 1, 1, model_size]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Normalize attention scores with sigmoid
        attention = self.sigmoid(scores)

        # Get the sum of attention scores along last dim and unsqueeze it (shape is [batch_size, num_heads, seq_len])
        sum_weights = torch.sum(attention, dim = -1).unsqueeze(-1)

        # Finally divide the normalized attention scores by the sum of all scores in the sequence
        attention_normalized = torch.div(attention, sum_weights)

        # Apply dropout to normalized attention scores
        attention_do = self.dropout(attention)

        # Get context vectors
        context = torch.matmul(attention_do, v)

        # Reshape context vectors to [batch_size, model_size, seq_len]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.head_size)

        # Output feedforward layer
        output = self.output_layer(context)

        return output, attention_normalized

class WordHead(nn.Module):

    def __init__(self, model_dim):
        super(WordHead, self).__init__()
        self.src = nn.Linear(model_dim, 2)
        self.trg = nn.Linear(model_dim, 2)

    def forward(self, x):
        src = self.src(x)
        trg = self.trg(x)
        return src, trg

class SequenceHead(nn.Module):

    def __init__(self, model_dim, nb_heads, attention_dropout, nb_metrics):
        super(SequenceHead, self).__init__()
        self.metric_tokens_vector = torch.arange(start = 0, end = nb_metrics, step = 1).to(torch.long)
        self.metric_embedding = nn.Embedding(nb_metrics, model_dim)
        self.metric_attn = MHA(num_heads = nb_heads, size = model_dim, dropout = attention_dropout)
        self.ter_out = nn.Linear(model_dim , 1)
        self.chrf_out = nn.Linear(model_dim , 1)
        self.bleu_out = nn.Linear(model_dim , 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask, src_mask = None, trg_mask = None):
        metrics_embedding = self.metric_embedding(self.metric_tokens_vector.to(x[0].device))
        metrics_embedding = metrics_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        metrics_attn, attention = self.metric_attn(k = x, v = x, q = metrics_embedding, mask = mask)
        ter = self.ter_out(metrics_attn[:, 0, :].squeeze(1)).squeeze(-1)
        chrf = self.chrf_out(metrics_attn[:, 1, :].squeeze(1)).squeeze(-1)
        bleu = self.bleu_out(metrics_attn[:, 2, :].squeeze(1)).squeeze(-1)
        return ter, chrf, bleu, attention

class MTQE(nn.Module):

    def __init__(self, model_name, attention_dropout = 0.0, nb_heads = 8, finetune = False):
        super(MTQE, self).__init__()
        if finetune: self.nb_metrics = 4
        else: self.nb_metrics = 3
        self.pretrained_model = XLMRobertaModel.from_pretrained(model_name, add_pooling_layer = False)
        model_dim = self.pretrained_model.get_input_embeddings().embedding_dim
        self.seq_head = SequenceHead(model_dim, nb_heads, attention_dropout, self.nb_metrics)
        self.heads_ff = nn.Linear(nb_heads, 1)
        self.metrics_ff = nn.Linear(model_dim + self.nb_metrics, 2)

    def forward(self, input_ids, attention_mask, special_tokens_mask):
        x = self.pretrained_model(\
            input_ids = input_ids, \
            attention_mask = attention_mask, \
        )[0]
        mask = ~((attention_mask.bool() & ~(special_tokens_mask.bool()))).unsqueeze(1)
        ter, chrf, bleu, attn = self.seq_head(x, mask)
        attn = self.heads_ff(attn.transpose(1, 3)).squeeze(-1)
        attn = torch.cat((x, attn), dim = -1)
        attn = self.metrics_ff(attn)
        return {"TER": ter, "CHRF": chrf, "BLEU": bleu, "TOKEN": attn.permute(0, 2, 1)}
