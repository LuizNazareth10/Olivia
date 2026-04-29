from __future__ import annotations

from collections import OrderedDict
import importlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.synthetic_hrv import generate_client_dataset
from src.models.bilstm_attention import BiLSTMAttention

try:
  fl = importlib.import_module("flwr")
except Exception:  # pragma: no cover
  fl = None


def get_parameters(model: nn.Module):
  return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
  params_dict = zip(model.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  model.load_state_dict(state_dict, strict=True)


class HrvClient((fl.client.NumPyClient if fl is not None else object)):
  def __init__(self, cid: int, local_epochs: int = 1):
    self.cid = cid
    self.local_epochs = local_epochs
    self.model = BiLSTMAttention()

    shift = (cid % 4) * 2.5
    x, y = generate_client_dataset(samples=256, seed=cid + 10, profile_shift=shift)
    ds = TensorDataset(torch.tensor(x), torch.tensor(y))
    self.loader = DataLoader(ds, batch_size=32, shuffle=True)

  def get_parameters(self, config):
    return get_parameters(self.model)

  def fit(self, parameters, config):
    set_parameters(self.model, parameters)

    opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    self.model.train()
    for _ in range(self.local_epochs):
      for xb, yb in self.loader:
        pred = self.model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return get_parameters(self.model), len(self.loader.dataset), {}

  def evaluate(self, parameters, config):
    set_parameters(self.model, parameters)
    criterion = nn.BCELoss()
    self.model.eval()

    losses = []
    accs = []
    with torch.no_grad():
      for xb, yb in self.loader:
        pred = self.model(xb)
        loss = criterion(pred, yb).item()
        acc = ((pred > 0.5) == (yb > 0.5)).float().mean().item()
        losses.append(loss)
        accs.append(acc)

    return float(np.mean(losses)), len(self.loader.dataset), {"acc": float(np.mean(accs))}


def client_fn(context: Any):
  if fl is None:
    raise RuntimeError("Flower (flwr) não está instalado neste ambiente.")

  cid = int(context.node_config["partition-id"])
  local_epochs = int(context.run_config.get("local-epochs", 1))
  return HrvClient(cid=cid, local_epochs=local_epochs).to_client()
