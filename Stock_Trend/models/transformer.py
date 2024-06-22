import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # Ensure positional encoding is on the same device as the input
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=8, num_encoder_layers=2, dropout=0.1, num_classes=2):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.input_layer = nn.Linear(input_dim, d_model)
        self.out_layer = nn.Linear(d_model, num_classes)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x = x.mean(dim=1)
        x = self.norm(x[:, -1, :])
        x = self.out_layer(x)
        return x


if __name__ == '__main__':
    # Initialize the model, loss function, and optimizer
    model = TransformerModel(input_dim=5)
    X = torch.randn(128, 7, 5)
    Y = model(X)
    print(Y.shape)
