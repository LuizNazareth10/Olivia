from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.synthetic_hrv import generate_client_dataset
from src.models.bilstm_attention import BiLSTMAttention


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--rounds", type=int, default=5)
  parser.add_argument("--clients", type=int, default=8)
  parser.add_argument("--local-epochs", type=int, default=2)
  return parser.parse_args()


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
  return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _fedavg(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
  avg = {}
  for key in states[0].keys():
    avg[key] = torch.stack([state[key] for state in states], dim=0).mean(dim=0)
  return avg


def _local_train(initial_state: dict[str, torch.Tensor], cid: int, local_epochs: int) -> dict[str, torch.Tensor]:
  model = BiLSTMAttention()
  model.load_state_dict(initial_state)

  shift = (cid % 4) * 2.5
  x, y = generate_client_dataset(samples=256, seed=cid + 100, profile_shift=shift)
  loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=32, shuffle=True)

  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.BCELoss()
  model.train()

  for _ in range(local_epochs):
    for xb, yb in loader:
      pred = model(xb)
      loss = criterion(pred, yb)
      opt.zero_grad()
      loss.backward()
      opt.step()

  return _clone_state_dict(model)


def main():
  args = parse_args()

  global_model = BiLSTMAttention()

  for _ in range(args.rounds):
    global_state = _clone_state_dict(global_model)
    local_states = []
    for cid in range(args.clients):
      local_states.append(_local_train(global_state, cid=cid, local_epochs=args.local_epochs))

    new_state = _fedavg(local_states)
    global_model.load_state_dict(new_state)

  Path("models").mkdir(parents=True, exist_ok=True)
  torch.save(global_model.state_dict(), "models/global_model.pt")

  dummy = torch.randn(1, 30, 4)
  torch.onnx.export(
    global_model,
    dummy,
    "models/global_model.onnx",
    input_names=["input"],
    output_names=["stress_probability"],
    dynamic_axes={"input": {0: "batch"}},
  )

  Path("models/global_model.tflite").write_bytes(
    b"Placeholder TFLite. Convert ONNX/PT in dedicated mobile export pipeline."
  )


if __name__ == "__main__":
  main()
