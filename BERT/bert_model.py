import numpy as np
import torch
from torch import nn

import config
from static_function import gelu, get_attn_pad_mask


class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.pos = None
        self.tok_embed = nn.Embedding(vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.maxlen, config.d_model)
        self.seg_embed = nn.Embedding(config.n_segments, config.d_model)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        self.pos = torch.arange(seq_len, dtype=torch.long)
        self.pos = self.pos.unsqueeze(0).expand_as(input_ids)
        embedding = self.tok_embed(input_ids) + self.pos_embed(self.pos) + self.seg_embed(segment_ids)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_pad):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(config.d_k)
        scores.masked_fill_(attn_pad, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(config.d_model, config.d_k * config.n_heads)
        self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads)
        self.W_V = nn.Linear(config.d_model, config.d_v * config.n_heads)

    def forward(self, Q, K, V, attn_pad):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, - 1, config.n_heads, config.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, - 1, config.n_heads, config.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, - 1, config.n_heads, config.d_v).transpose(1, 2)
        attn_pad = attn_pad.unsqueeze(1).repeat(1, config.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_pad)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, config.n_heads*config.d_v)
        output = nn.Linear(config.n_heads * config.d_v, config.d_model)(context)
        return nn.LayerNorm(config.d_model)(output + residual), attn


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_pad):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_pad)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class bert(nn.Module):
    def __init__(self, vocab_size):
        super(bert, self).__init__()
        self.embedding = Embedding(vocab_size)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.n_layers)])
        self.fc = nn.Linear(config.d_model, config.d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, 2)

        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, inputs):
        input_ids, segment_ids, masked_pos = inputs[0], inputs[1], inputs[2]
        input_ = self.embedding(input_ids, segment_ids)
        enc_self_attn_pad = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(input_, enc_self_attn_pad)
        h_pooled = self.activ1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_clsf
