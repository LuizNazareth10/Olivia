import torch
from torch import nn


class BiLSTMAttention(nn.Module):
  def __init__(self, input_dim: int = 4, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.2):
    super().__init__()
    self.bilstm = nn.LSTM(
      input_size=input_dim,
      hidden_size=hidden_dim,
      num_layers=layers,
      batch_first=True,
      bidirectional=True,
      dropout=dropout if layers > 1 else 0.0,
    )

    self.attention = nn.Sequential(
      nn.Linear(hidden_dim * 2, hidden_dim),
      nn.Tanh(),
      nn.Linear(hidden_dim, 1),
    )

    self.classifier = nn.Sequential(
      nn.Linear(hidden_dim * 2, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, 1),
      nn.Sigmoid(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    lstm_out, _ = self.bilstm(x)
    attn_scores = self.attention(lstm_out)
    attn_weights = torch.softmax(attn_scores, dim=1)
    context = torch.sum(attn_weights * lstm_out, dim=1)
    out = self.classifier(context)
    return out.squeeze(-1)
