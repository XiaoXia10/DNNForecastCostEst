#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM with residual connections, LayerNorm, and 4 MC dropout points.
Registered as "lstm" in the model registry.
"""

import torch
import torch.nn as nn
from framework.model_registry import register


@register("lstm")
class BayesianLSTM(nn.Module):

    """
    LSTM with residuals and xavier weight init
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        residual: bool = True,
        ) -> None:

        super().__init__()

        self.residual = residual and input_size == output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.input_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.hidden_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for fc in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.input_dropout(x))
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.act(out)
        out = self.hidden_dropout(out)
        out = self.fc2(out)

        if self.residual:
            out = out + x

        return out
