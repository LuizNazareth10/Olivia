from __future__ import annotations

import numpy as np


def generate_client_dataset(
  samples: int = 256,
  seq_len: int = 30,
  seed: int | None = None,
  profile_shift: float = 0.0,
):
  rng = np.random.default_rng(seed)

  x = np.zeros((samples, seq_len, 4), dtype=np.float32)
  y = np.zeros((samples,), dtype=np.float32)

  for i in range(samples):
    base_rmssd = rng.normal(42 + profile_shift, 12, seq_len)
    base_sdnn = rng.normal(48 + profile_shift, 13, seq_len)
    base_pnn50 = rng.normal(22 + profile_shift, 10, seq_len)
    base_hr = rng.normal(74 - profile_shift * 0.25, 9, seq_len)

    stress_prob = np.clip(
      0.65 * (1 - (np.mean(base_rmssd + base_sdnn + base_pnn50) / 150))
      + 0.35 * (np.mean(base_hr) / 120),
      0,
      1,
    )
    label = 1.0 if rng.random() < stress_prob else 0.0

    if label > 0.5:
      base_rmssd -= rng.normal(10, 5, seq_len)
      base_sdnn -= rng.normal(8, 4, seq_len)
      base_pnn50 -= rng.normal(6, 3, seq_len)
      base_hr += rng.normal(7, 4, seq_len)

    x[i, :, 0] = base_rmssd
    x[i, :, 1] = base_sdnn
    x[i, :, 2] = base_pnn50
    x[i, :, 3] = base_hr
    y[i] = label

  x = _normalize(x)
  return x.astype(np.float32), y.astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
  mean = x.mean(axis=(0, 1), keepdims=True)
  std = x.std(axis=(0, 1), keepdims=True) + 1e-6
  return (x - mean) / std
