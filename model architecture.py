# model_architecture.py

import torch
import torch.nn as nn

class PixelLSTM(nn.Module):
    def __init__(self, input_size, static_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.static_to_hidden = nn.Linear(static_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, x_static, return_hidden=False):
        h0 = self.static_to_hidden(x_static).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = h0.clone()
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.output_layer(lstm_out).squeeze(-1)
        return (out, lstm_out) if return_hidden else out
