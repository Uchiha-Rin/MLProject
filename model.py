import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[:len(pe[:, 0::2][0])])
        pe[:, 1::2] = torch.cos(position * div_term[:len(pe[:, 1::2][0])])
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransAm(nn.Module):
    def __init__(self, feature_size=7, num_layers=3, hidden_dim=128, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.feature_size = feature_size
        self.src_mask = None
        # Encoder
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.feedforward = FeedForwardLayer(feature_size, hidden_dim, dropout=dropout)

        self.decoder = nn.Linear(feature_size, feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        device = src.device
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            # print('a',src.size())
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # Encoder
        src = self.pos_encoder(src)
        encoder_memory = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.feedforward(encoder_memory)

        output = output.permute(1, 0, 2)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask