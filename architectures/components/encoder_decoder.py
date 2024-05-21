import numpy as np
import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .mlp import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model)
        self.cross_attn = MultiHeadedAttention(num_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
        
        
class Transformer(nn.Module):
    def __init__(self, src_feats, trg_feats, out_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.embed_src = nn.Linear(src_feats, d_model)
        self.embed_trg = nn.Linear(trg_feats, d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.generate_output = nn.Linear(d_model, out_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, seq, mask_val = -99):
        src_mask = (src != mask_val).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != mask_val).unsqueeze(1).unsqueeze(3)        
        
        seq_length = src.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        src_mask = src_mask & nopeak_mask
        
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, mask_val = -99):
        src_mask, tgt_mask = self.generate_mask(src, tgt, mask_val=mask_val)
        src_embedded = self.dropout(self.encoder_embedding(src))
        tgt_embedded = self.dropout(self.decoder_embedding(tgt))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
