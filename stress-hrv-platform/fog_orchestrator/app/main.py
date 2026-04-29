import base64
import hmac
import hashlib
import io
import math
import os
import random
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
  import torch
  from torch import nn
except Exception:  # pragma: no cover
  torch = None
  nn = None

CLOUD_URL = os.getenv("CLOUD_URL", "http://localhost:8080")
FOG_ID = os.getenv("FOG_ID", "fog-local-1")

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "federated_local.db"
MODEL_PATH = BASE_DIR / "models" / "global_model.pt"
PERSONAL_MODELS_DIR = BASE_DIR / "models" / "personal"

MIN_CLIENTS_PER_ROUND = 2
MIN_SAMPLES_PER_CLIENT = 8
MAX_SAMPLES_PER_CLIENT = 128
LOCAL_EPOCHS = 2
PERSONAL_TRAIN_INTERVAL = 50

_LOCK = threading.Lock()
_EVAL_CACHE: dict[str, Any] = {"ts": 0.0, "summary": None}

PROFILE_SPECS: dict[str, dict[str, Any]] = {
  "endurance_athlete": {
    "label": "Perfil 1 — Atleta de endurance",
    "user_id": "profile_endurance",
    "device_id": "device_endurance",
    "age_range": (25, 35),
    "rmssd_range": (60.0, 110.0),
    "sdnn_range": (65.0, 125.0),
    "pnn50_range": (28.0, 55.0),
    "mean_hr_range": (40.0, 55.0),
    "stress_bias": 0.12,
  },
  "active_recreational": {
    "label": "Perfil 2 — Adulto ativo recreacional",
    "user_id": "profile_active",
    "device_id": "device_active",
    "age_range": (25, 45),
    "rmssd_range": (35.0, 60.0),
    "sdnn_range": (38.0, 72.0),
    "pnn50_range": (16.0, 35.0),
    "mean_hr_range": (55.0, 70.0),
    "stress_bias": 0.28,
  },
  "sedentary_overweight": {
    "label": "Perfil 3 — Sedentário com sobrepeso",
    "user_id": "profile_sedentary",
    "device_id": "device_sedentary",
    "age_range": (30, 50),
    "rmssd_range": (15.0, 30.0),
    "sdnn_range": (18.0, 38.0),
    "pnn50_range": (4.0, 16.0),
    "mean_hr_range": (75.0, 95.0),
    "stress_bias": 0.72,
  },
  "chronic_stress": {
    "label": "Perfil 4 — Estresse crônico",
    "user_id": "profile_chronic_stress",
    "device_id": "device_chronic_stress",
    "age_range": (28, 52),
    "rmssd_range": (10.0, 25.0),
    "sdnn_range": (14.0, 32.0),
    "pnn50_range": (2.0, 12.0),
    "mean_hr_range": (70.0, 90.0),
    "stress_bias": 0.82,
  },
  "healthy_active_senior": {
    "label": "Perfil 5 — Idoso saudável ativo",
    "user_id": "profile_senior",
    "device_id": "device_senior",
    "age_range": (65, 75),
    "rmssd_range": (20.0, 40.0),
    "sdnn_range": (22.0, 48.0),
    "pnn50_range": (8.0, 22.0),
    "mean_hr_range": (60.0, 75.0),
    "stress_bias": 0.42,
  },
}


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


app = FastAPI(title="Fog Orchestrator", version="2.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

_MODEL = None
if torch is not None:
  try:
    _MODEL = BiLSTMAttention()
    _MODEL.eval()
    if MODEL_PATH.exists():
      state = torch.load(MODEL_PATH, map_location="cpu")
      _MODEL.load_state_dict(state)
  except Exception:
    _MODEL = None


class RiskPayload(BaseModel):
  user_id: str
  device_id: str
  metrics: dict[str, float]
  local_probability: float
  sampling_hz: float


class SimulateProfilesPayload(BaseModel):
  samples_per_profile: int = Field(default=250, ge=50, le=5000)
  sampling_hz: float = Field(default=30.0, gt=0)
  seed: int = Field(default=20260225)
  run_global_training: bool = True


class RegisterPayload(BaseModel):
  name: str = Field(min_length=2, max_length=120)
  email: str = Field(min_length=5, max_length=160)
  password: str = Field(min_length=6, max_length=200)


class LoginPayload(BaseModel):
  email: str = Field(min_length=5, max_length=160)
  password: str = Field(min_length=6, max_length=200)


class LogoutPayload(BaseModel):
  token: str = Field(min_length=16, max_length=256)


def _get_conn() -> sqlite3.Connection:
  DB_DIR.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn


def _init_db() -> None:
  with _get_conn() as conn:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS auth_sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY(user_id) REFERENCES users(id)
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS captures_local (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        device_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        sampling_hz REAL NOT NULL,
        rmssd REAL NOT NULL,
        sdnn REAL NOT NULL,
        pnn50 REAL NOT NULL,
        mean_hr REAL NOT NULL,
        local_probability REAL NOT NULL,
        final_probability REAL NOT NULL,
        high_stress INTEGER NOT NULL,
        trained_round INTEGER
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS device_model_state (
        device_id TEXT PRIMARY KEY,
        samples_seen INTEGER NOT NULL DEFAULT 0,
        last_personal_version INTEGER NOT NULL DEFAULT 0,
        last_global_version INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS local_rounds (
        round_version INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        clients INTEGER NOT NULL,
        samples INTEGER NOT NULL,
        mae REAL,
        rmse REAL,
        smape REAL,
        mase REAL
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS personal_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        device_id TEXT NOT NULL,
        personal_version INTEGER NOT NULL,
        samples INTEGER NOT NULL,
        mae REAL,
        rmse REAL
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
      """
    )
    conn.execute("INSERT OR IGNORE INTO metadata(key, value) VALUES('model_version', '0')")
    salt_row = conn.execute("SELECT value FROM metadata WHERE key='anon_salt'").fetchone()
    if salt_row is None:
      conn.execute(
        "INSERT INTO metadata(key, value) VALUES('anon_salt', ?)",
        (os.urandom(16).hex(),),
      )
    conn.commit()


def _norm_email(email: str) -> str:
  return email.strip().lower()


def _hash_password(password: str, salt: str) -> str:
  return hashlib.pbkdf2_hmac(
    "sha256",
    password.encode("utf-8"),
    bytes.fromhex(salt),
    120_000,
  ).hex()


def _new_session_token() -> str:
  return secrets.token_urlsafe(32)


def _session_expiration_iso(hours: int = 24 * 14) -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + hours * 3600))


def _auth_user_from_token(conn: sqlite3.Connection, token: str) -> sqlite3.Row | None:
  row = conn.execute(
    """
    SELECT u.id, u.name, u.email, s.token, s.expires_at
    FROM auth_sessions s
    JOIN users u ON u.id = s.user_id
    WHERE s.token = ? AND s.active = 1 AND u.active = 1 AND s.expires_at > CURRENT_TIMESTAMP
    """,
    (token,),
  ).fetchone()
  return row


def _safe_number(value: float | None) -> float | None:
  if value is None:
    return None
  num = float(value)
  if math.isnan(num) or math.isinf(num):
    return None
  return num


def _get_model_version(conn: sqlite3.Connection) -> int:
  row = conn.execute("SELECT value FROM metadata WHERE key='model_version'").fetchone()
  if row is None:
    conn.execute("INSERT INTO metadata(key, value) VALUES('model_version', '0')")
    conn.commit()
    return 0
  return int(row["value"])


def _set_model_version(conn: sqlite3.Connection, version: int) -> None:
  conn.execute("UPDATE metadata SET value=? WHERE key='model_version'", (str(version),))


def _get_anon_salt(conn: sqlite3.Connection) -> str:
  row = conn.execute("SELECT value FROM metadata WHERE key='anon_salt'").fetchone()
  if row is None:
    salt = os.urandom(16).hex()
    conn.execute("INSERT INTO metadata(key, value) VALUES('anon_salt', ?)", (salt,))
    conn.commit()
    return salt
  return str(row["value"])


def _participant_alias(conn: sqlite3.Connection, device_id: str) -> str:
  salt = _get_anon_salt(conn)
  digest = hashlib.sha256(f"{salt}:{device_id}".encode("utf-8")).hexdigest()[:10].upper()
  return f"C-{digest}"


def _heuristic_probability(metrics: dict[str, float]) -> float:
  rmssd = float(metrics.get("rmssd", 0.0))
  sdnn = float(metrics.get("sdnn", 0.0))
  pnn50 = float(metrics.get("pnn50", 0.0))
  mean_hr = float(metrics.get("mean_hr", 0.0))

  hrv_score = (rmssd + sdnn + pnn50) / 3.0
  normalized_hrv = max(0.0, min(1.0, hrv_score / 80.0))
  normalized_hr = max(0.0, min(1.0, mean_hr / 120.0))
  probability = 0.65 * (1.0 - normalized_hrv) + 0.35 * normalized_hr
  return max(0.0, min(1.0, probability))


def _sample_profile_metrics(spec: dict[str, Any], rng: random.Random) -> dict[str, float]:
  rmssd = rng.uniform(*spec["rmssd_range"])
  sdnn = rng.uniform(*spec["sdnn_range"])
  pnn50 = rng.uniform(*spec["pnn50_range"])
  mean_hr = rng.uniform(*spec["mean_hr_range"])

  hrv_jitter = rng.uniform(-0.08, 0.08)
  hr_jitter = rng.uniform(-3.0, 3.0)

  rmssd = max(1.0, rmssd * (1.0 + hrv_jitter))
  sdnn = max(1.0, sdnn * (1.0 + hrv_jitter))
  pnn50 = max(0.0, pnn50 * (1.0 + hrv_jitter))
  mean_hr = max(30.0, mean_hr + hr_jitter)

  return {
    "rmssd": float(rmssd),
    "sdnn": float(sdnn),
    "pnn50": float(pnn50),
    "mean_hr": float(mean_hr),
  }


def _global_model_probability(metrics: dict[str, float]) -> float | None:
  if _MODEL is None or torch is None:
    return None

  rmssd = float(metrics.get("rmssd", 0.0))
  sdnn = float(metrics.get("sdnn", 0.0))
  pnn50 = float(metrics.get("pnn50", 0.0))
  mean_hr = float(metrics.get("mean_hr", 0.0))

  features = torch.tensor([[[rmssd, sdnn, pnn50, mean_hr]] * 30], dtype=torch.float32)
  with torch.no_grad():
    prob = float(_MODEL(features).item())
  return max(0.0, min(1.0, prob))


def _personal_model_path(device_id: str) -> Path:
  safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in device_id)
  return PERSONAL_MODELS_DIR / f"{safe}.pt"


def _personal_model_probability(device_id: str, metrics: dict[str, float]) -> float | None:
  if torch is None:
    return None

  path = _personal_model_path(device_id)
  if not path.exists():
    return None

  try:
    model = BiLSTMAttention()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    rmssd = float(metrics.get("rmssd", 0.0))
    sdnn = float(metrics.get("sdnn", 0.0))
    pnn50 = float(metrics.get("pnn50", 0.0))
    mean_hr = float(metrics.get("mean_hr", 0.0))
    features = torch.tensor([[[rmssd, sdnn, pnn50, mean_hr]] * 30], dtype=torch.float32)
    with torch.no_grad():
      prob = float(model(features).item())
    return max(0.0, min(1.0, prob))
  except Exception:
    return None


def _train_personal_model(conn: sqlite3.Connection, device_id: str, sample_count: int) -> dict[str, Any]:
  if torch is None:
    return {"trained": False, "reason": "torch-unavailable"}

  rows = conn.execute(
    """
    SELECT * FROM captures_local
    WHERE device_id = ?
    ORDER BY id ASC
    LIMIT ?
    """,
    (device_id, MAX_SAMPLES_PER_CLIENT),
  ).fetchall()
  if len(rows) < PERSONAL_TRAIN_INTERVAL:
    return {"trained": False, "reason": "insufficient-samples"}

  base_model = BiLSTMAttention()
  if _MODEL is not None:
    base_model.load_state_dict(_MODEL.state_dict())
  base_model.train()

  x, y = _build_training_tensors(rows)
  if x is None or y is None or len(x) < 8:
    return {"trained": False, "reason": "insufficient-sequences"}
  opt = torch.optim.Adam(base_model.parameters(), lr=8e-4)
  criterion = torch.nn.BCELoss()

  for _ in range(max(LOCAL_EPOCHS, 2)):
    pred = base_model(x)
    loss = criterion(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

  base_model.eval()
  with torch.no_grad():
    pred = base_model(x).cpu().numpy()
  y_np = y.cpu().numpy()

  mae = float(np.mean(np.abs(y_np - pred)))
  rmse = float(np.sqrt(np.mean((y_np - pred) ** 2)))

  PERSONAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
  model_version = int(sample_count // PERSONAL_TRAIN_INTERVAL)
  torch.save(base_model.state_dict(), _personal_model_path(device_id))

  conn.execute(
    """
    INSERT INTO device_model_state(device_id, samples_seen, last_personal_version, last_global_version, updated_at)
    VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(device_id) DO UPDATE SET
      samples_seen=excluded.samples_seen,
      last_personal_version=excluded.last_personal_version,
      updated_at=CURRENT_TIMESTAMP
    """,
    (device_id, int(sample_count), model_version, _get_model_version(conn)),
  )
  conn.execute(
    """
    INSERT INTO personal_rounds(device_id, personal_version, samples, mae, rmse)
    VALUES(?, ?, ?, ?, ?)
    """,
    (device_id, model_version, int(sample_count), _safe_number(mae), _safe_number(rmse)),
  )

  return {
    "trained": True,
    "device_id": device_id,
    "personal_version": model_version,
    "samples": int(sample_count),
    "mae": _safe_number(mae),
    "rmse": _safe_number(rmse),
  }


def _build_training_tensors(rows: list[sqlite3.Row]):
  seq_len = 30
  horizon = 1
  if len(rows) < seq_len + horizon:
    return None, None

  feat = np.array(
    [[float(r["rmssd"]), float(r["sdnn"]), float(r["pnn50"]), float(r["mean_hr"])] for r in rows],
    dtype=np.float32,
  )
  target = np.array([float(r["final_probability"]) for r in rows], dtype=np.float32)
  target = np.clip(target, 0.0, 1.0)

  x_list = []
  y_list = []
  for i in range(seq_len, len(rows) - horizon + 1):
    x_list.append(feat[i - seq_len:i])
    y_list.append(target[i + horizon - 1])

  if len(x_list) < 8:
    return None, None

  x = torch.tensor(np.array(x_list, dtype=np.float32), dtype=torch.float32)
  y = torch.tensor(np.array(y_list, dtype=np.float32), dtype=torch.float32)
  return x, y


def _clone_state_dict(model) -> dict[str, Any]:
  return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _fedavg(states: list[dict[str, Any]]) -> dict[str, Any]:
  avg = {}
  for key in states[0].keys():
    avg[key] = torch.stack([state[key] for state in states], dim=0).mean(dim=0)
  return avg


def _local_train(initial_state: dict[str, Any], rows: list[sqlite3.Row]) -> tuple[dict[str, Any], dict[str, float]] | None:
  model = BiLSTMAttention()
  model.load_state_dict(initial_state)
  model.train()

  x, y = _build_training_tensors(rows)
  if x is None or y is None or len(x) < 8:
    return None
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.BCELoss()

  for _ in range(LOCAL_EPOCHS):
    pred = model(x)
    loss = criterion(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

  model.eval()
  with torch.no_grad():
    pred = model(x).cpu().numpy()
  y_np = y.cpu().numpy()

  mae = float(np.mean(np.abs(y_np - pred)))
  rmse = float(np.sqrt(np.mean((y_np - pred) ** 2)))
  denom = np.abs(y_np) + np.abs(pred)
  denom = np.where(denom == 0, 1e-8, denom)
  smape = float(np.mean(2.0 * np.abs(pred - y_np) / denom))
  if len(y_np) > 1:
    scale = float(np.mean(np.abs(np.diff(y_np))))
    if scale == 0:
      scale = 1e-8
    mase = float(np.mean(np.abs(y_np - pred)) / scale)
  else:
    mase = float("nan")

  return _clone_state_dict(model), {"mae": mae, "rmse": rmse, "smape": smape, "mase": mase}


def _push_model_update_to_cloud(
  round_version: int,
  clients: int,
  samples: int,
  metrics: dict[str, float],
  participant_aliases: list[str],
) -> None:
  if _MODEL is None or torch is None:
    return
  try:
    buffer = io.BytesIO()
    torch.save(_MODEL.state_dict(), buffer)
    payload = {
      "fog_id": FOG_ID,
      "round_version": round_version,
      "clients": clients,
      "samples": samples,
      "participant_aliases": sorted(participant_aliases),
      "cohort_key": (
        "COHORT-"
        + hashlib.sha256("|".join(sorted(participant_aliases)).encode("utf-8")).hexdigest()[:12].upper()
      ) if participant_aliases else None,
      "metrics": {
        "mae": _safe_number(metrics.get("mae")),
        "rmse": _safe_number(metrics.get("rmse")),
        "smape": _safe_number(metrics.get("smape")),
        "mase": _safe_number(metrics.get("mase")),
      },
      "state_b64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }
    with httpx.Client(timeout=15.0) as client:
      client.post(f"{CLOUD_URL}/federated/model/push", json=payload)
  except Exception:
    pass


def _try_local_federated_round(conn: sqlite3.Connection) -> dict[str, Any]:
  if _MODEL is None or torch is None:
    return {"trained": False, "reason": "model-unavailable"}

  device_rows = conn.execute(
    """
    SELECT device_id, COUNT(*) AS cnt
    FROM captures_local
    WHERE trained_round IS NULL
    GROUP BY device_id
    HAVING cnt >= ?
    """,
    (MIN_SAMPLES_PER_CLIENT,),
  ).fetchall()

  if len(device_rows) < MIN_CLIENTS_PER_ROUND:
    return {"trained": False, "reason": "insufficient-clients", "eligible_clients": len(device_rows)}

  current_version = _get_model_version(conn)
  next_version = current_version + 1

  global_state = _clone_state_dict(_MODEL)
  local_states = []
  used_capture_ids: list[int] = []
  metric_list = []
  trained_device_ids: list[str] = []

  for row in device_rows:
    device_id = row["device_id"]
    rows = conn.execute(
      """
      SELECT * FROM captures_local
      WHERE device_id = ? AND trained_round IS NULL
      ORDER BY id ASC
      LIMIT ?
      """,
      (device_id, MAX_SAMPLES_PER_CLIENT),
    ).fetchall()
    if len(rows) < MIN_SAMPLES_PER_CLIENT:
      continue

    local_result = _local_train(global_state, rows)
    if local_result is None:
      continue
    state, metric = local_result
    local_states.append(state)
    metric_list.append(metric)
    used_capture_ids.extend([int(r["id"]) for r in rows])
    trained_device_ids.append(device_id)

  if len(local_states) < MIN_CLIENTS_PER_ROUND:
    return {"trained": False, "reason": "insufficient-clients-after-filter", "eligible_clients": len(local_states)}

  new_state = _fedavg(local_states)
  _MODEL.load_state_dict(new_state)
  _MODEL.eval()

  MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
  torch.save(_MODEL.state_dict(), MODEL_PATH)

  placeholders = ",".join(["?"] * len(used_capture_ids))
  conn.execute(
    f"UPDATE captures_local SET trained_round = ? WHERE id IN ({placeholders})",
    (next_version, *used_capture_ids),
  )

  mean_metrics = {
    "mae": float(np.mean([m["mae"] for m in metric_list])) if metric_list else float("nan"),
    "rmse": float(np.mean([m["rmse"] for m in metric_list])) if metric_list else float("nan"),
    "smape": float(np.mean([m["smape"] for m in metric_list])) if metric_list else float("nan"),
    "mase": float(np.mean([m["mase"] for m in metric_list])) if metric_list else float("nan"),
  }

  conn.execute(
    """
    INSERT INTO local_rounds(round_version, clients, samples, mae, rmse, smape, mase)
    VALUES(?, ?, ?, ?, ?, ?, ?)
    """,
    (
      next_version,
      len(local_states),
      len(used_capture_ids),
      _safe_number(mean_metrics["mae"]),
      _safe_number(mean_metrics["rmse"]),
      _safe_number(mean_metrics["smape"]),
      _safe_number(mean_metrics["mase"]),
    ),
  )

  for row in device_rows:
    device_id = row["device_id"]
    sample_count = conn.execute(
      "SELECT COUNT(*) AS cnt FROM captures_local WHERE device_id = ?",
      (device_id,),
    ).fetchone()["cnt"]
    personal_version = conn.execute(
      "SELECT last_personal_version FROM device_model_state WHERE device_id = ?",
      (device_id,),
    ).fetchone()
    last_personal_version = int(personal_version["last_personal_version"]) if personal_version is not None else 0
    conn.execute(
      """
      INSERT INTO device_model_state(device_id, samples_seen, last_personal_version, last_global_version, updated_at)
      VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(device_id) DO UPDATE SET
        samples_seen=excluded.samples_seen,
        last_personal_version=excluded.last_personal_version,
        last_global_version=excluded.last_global_version,
        updated_at=CURRENT_TIMESTAMP
      """,
      (device_id, int(sample_count), last_personal_version, next_version),
    )

  _set_model_version(conn, next_version)
  conn.commit()

  _push_model_update_to_cloud(
    round_version=next_version,
    clients=len(local_states),
    samples=len(used_capture_ids),
    metrics=mean_metrics,
    participant_aliases=[_participant_alias(conn, device_id) for device_id in trained_device_ids],
  )

  return {
    "trained": True,
    "round_version": next_version,
    "clients": len(local_states),
    "samples": len(used_capture_ids),
    "metrics": {k: _safe_number(v) for k, v in mean_metrics.items()},
  }


def _device_rows(conn: sqlite3.Connection, device_id: str) -> list[sqlite3.Row]:
  return conn.execute(
    """
    SELECT id, created_at, rmssd, sdnn, pnn50, mean_hr, final_probability
    FROM captures_local
    WHERE device_id = ?
    ORDER BY id ASC
    """,
    (device_id,),
  ).fetchall()


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  denom = np.abs(y_true) + np.abs(y_pred)
  denom = np.where(denom == 0, 1e-8, denom)
  return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _mase(y_true: np.ndarray, y_pred: np.ndarray, in_sample: np.ndarray) -> float:
  if len(in_sample) < 2:
    return float("nan")
  scale = np.mean(np.abs(np.diff(in_sample)))
  if scale == 0:
    scale = 1e-8
  return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, in_sample: np.ndarray) -> dict[str, float | None]:
  mae = float(np.mean(np.abs(y_true - y_pred)))
  rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
  return {
    "mae": _safe_number(mae),
    "rmse": _safe_number(rmse),
    "smape": _safe_number(_smape(y_true, y_pred)),
    "mase": _safe_number(_mase(y_true, y_pred, in_sample)),
  }


def _build_sequences(rows: list[sqlite3.Row], horizon: int, seq_len: int = 30):
  if len(rows) < seq_len + horizon + 5:
    return None

  feat = np.array(
    [[float(r["rmssd"]), float(r["sdnn"]), float(r["pnn50"]), float(r["mean_hr"])] for r in rows],
    dtype=np.float32,
  )
  target = np.array([float(r["final_probability"]) for r in rows], dtype=np.float32)
  stamps = [str(r["created_at"]) for r in rows]

  x_list, y_list, t_list = [], [], []
  naive_list = []
  for i in range(seq_len, len(rows) - horizon + 1):
    x_list.append(feat[i - seq_len:i])
    y_list.append(target[i + horizon - 1])
    t_list.append(stamps[i + horizon - 1])
    naive_list.append(target[i - 1])

  x = np.array(x_list, dtype=np.float32)
  y = np.array(y_list, dtype=np.float32)
  naive = np.array(naive_list, dtype=np.float32)
  if len(x) < 20:
    return None

  split = max(int(len(x) * 0.7), 10)
  split = min(split, len(x) - 5)
  x_train, y_train = x[:split], y[:split]
  x_test, y_test = x[split:], y[split:]
  naive_train = naive[:split]
  naive_test = naive[split:]
  t_test = t_list[split:]

  mean = x_train.mean(axis=(0, 1), keepdims=True)
  std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
  x_train = (x_train - mean) / std
  x_test = (x_test - mean) / std

  return x_train, y_train, x_test, y_test, naive_train, naive_test, t_test


def _load_eval_model(device_id: str) -> tuple[Any | None, str]:
  if torch is None:
    return None, "unavailable"

  model = BiLSTMAttention(input_dim=4, hidden_dim=64, layers=2, dropout=0.2)
  personal_path = _personal_model_path(device_id)
  try:
    if personal_path.exists():
      model.load_state_dict(torch.load(personal_path, map_location="cpu"))
      model.eval()
      return model, "personal"
    if _MODEL is not None:
      model.load_state_dict(_MODEL.state_dict())
      model.eval()
      return model, "global"
  except Exception:
    return None, "unavailable"

  return None, "unavailable"


def _evaluate_device_horizon(rows: list[sqlite3.Row], horizon: int, device_id: str) -> dict[str, Any] | None:
  if torch is None:
    return None
  data = _build_sequences(rows, horizon=horizon)
  if data is None:
    return None

  x_train, y_train, x_test, y_test, naive_train, naive_test, t_test = data
  if len(x_test) == 0:
    return None

  model, source = _load_eval_model(device_id)
  if model is None:
    return None
  model.train()

  x_train_t = torch.tensor(x_train, dtype=torch.float32)
  y_train_t = torch.tensor(y_train, dtype=torch.float32)
  x_test_t = torch.tensor(x_test, dtype=torch.float32)

  opt = torch.optim.Adam(model.parameters(), lr=8e-4)
  criterion = torch.nn.BCELoss()

  for _ in range(12):
    pred = model(x_train_t)
    loss = criterion(pred, y_train_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

  model.eval()
  with torch.no_grad():
    pred_train = model(x_train_t).cpu().numpy()
    pred_test = model(x_test_t).cpu().numpy()

  pred_test = np.clip(pred_test, 0.0, 1.0)
  pred_train = np.clip(pred_train, 0.0, 1.0)

  naive_train = np.clip(naive_train, 0.0, 1.0)
  naive_test = np.clip(naive_test, 0.0, 1.0)
  model_metrics = _eval_metrics(y_test, pred_test, y_train)
  naive_metrics = _eval_metrics(y_test, naive_test, y_train)

  model_mae = model_metrics.get("mae")
  naive_mae = naive_metrics.get("mae")
  use_model = (
    model_mae is not None and naive_mae is not None and float(model_mae) <= float(naive_mae)
  )

  chosen_pred_test = pred_test if use_model else naive_test
  chosen_pred_train = pred_train if use_model else naive_train
  selected_name = source if use_model else "naive-last-value"

  resid_std = float(np.std(y_train - chosen_pred_train))
  low = np.clip(chosen_pred_test - 1.96 * resid_std, 0.0, 1.0)
  high = np.clip(chosen_pred_test + 1.96 * resid_std, 0.0, 1.0)
  coverage = float(np.mean((y_test >= low) & (y_test <= high)))
  avg_width = float(np.mean(high - low))

  return {
    "horizon": horizon,
    "metrics": _eval_metrics(y_test, chosen_pred_test, y_train),
    "model_source": source,
    "selected_predictor": selected_name,
    "baseline_naive": naive_metrics,
    "series": {
      "timestamps": t_test,
      "actual": y_test.tolist(),
      "forecast": chosen_pred_test.tolist(),
      "lower": low.tolist(),
      "upper": high.tolist(),
    },
    "interval": {"coverage": _safe_number(coverage), "avg_width": _safe_number(avg_width)},
  }


def _backtesting_windows(actual: list[float], forecast: list[float], windows: int = 5) -> list[dict[str, Any]]:
  n = min(len(actual), len(forecast))
  if n < windows:
    return []
  w = max(1, n // windows)
  result = []
  for i in range(windows):
    s = i * w
    e = n if i == windows - 1 else min((i + 1) * w, n)
    if e - s < 1:
      continue
    a = np.array(actual[s:e], dtype=np.float32)
    p = np.array(forecast[s:e], dtype=np.float32)
    result.append(
      {
        "window": i + 1,
        "mae": _safe_number(float(np.mean(np.abs(a - p)))),
        "rmse": _safe_number(float(np.sqrt(np.mean((a - p) ** 2)))),
      }
    )
  return result


def _compute_eval_summary(force: bool = False) -> dict[str, Any]:
  now = time.time()
  if not force and _EVAL_CACHE.get("summary") is not None and now - float(_EVAL_CACHE.get("ts", 0)) < 30:
    return _EVAL_CACHE["summary"]

  with _get_conn() as conn:
    devices = [row["device_id"] for row in conn.execute("SELECT DISTINCT device_id FROM captures_local").fetchall()]
    per_device = []
    for d in devices:
      rows = _device_rows(conn, d)
      if len(rows) < 40:
        continue
      horizons = {}
      for h in [1, 7, 30]:
        ev = _evaluate_device_horizon(rows, h, d)
        if ev is not None:
          horizons[f"t+{h}"] = ev["metrics"]
      if not horizons:
        continue
      overall = horizons.get("t+1") or next(iter(horizons.values()))
      per_device.append({"device_id": d, "samples": len(rows), "overall": overall, "horizons": horizons})

    if per_device:
      mae_vals = [d["overall"]["mae"] for d in per_device if d["overall"]["mae"] is not None]
      rmse_vals = [d["overall"]["rmse"] for d in per_device if d["overall"]["rmse"] is not None]
      smape_vals = [d["overall"]["smape"] for d in per_device if d["overall"]["smape"] is not None]
      mase_vals = [d["overall"]["mase"] for d in per_device if d["overall"]["mase"] is not None]
      global_overall = {
        "mae": _safe_number(float(np.mean(mae_vals))) if mae_vals else None,
        "rmse": _safe_number(float(np.mean(rmse_vals))) if rmse_vals else None,
        "smape": _safe_number(float(np.mean(smape_vals))) if smape_vals else None,
        "mase": _safe_number(float(np.mean(mase_vals))) if mase_vals else None,
      }
    else:
      global_overall = {"mae": None, "rmse": None, "smape": None, "mase": None}

    out = {
      "generated_at": int(now),
      "devices_with_eval": len(per_device),
      "global_overall": global_overall,
      "devices": per_device,
    }

  _EVAL_CACHE["ts"] = now
  _EVAL_CACHE["summary"] = out
  return out


@app.get("/health")
def health() -> dict[str, str]:
  return {"status": "ok", "layer": "fog"}


@app.post("/auth/register")
def auth_register(payload: RegisterPayload):
  email = _norm_email(payload.email)
  with _LOCK:
    with _get_conn() as conn:
      exists = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
      if exists is not None:
        return {"ok": False, "reason": "email-already-registered"}

      salt = secrets.token_hex(16)
      password_hash = _hash_password(payload.password, salt)
      conn.execute(
        """
        INSERT INTO users(name, email, password_hash, password_salt)
        VALUES(?, ?, ?, ?)
        """,
        (payload.name.strip(), email, password_hash, salt),
      )
      user_id = int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])

      token = _new_session_token()
      expires_at = _session_expiration_iso()
      conn.execute(
        """
        INSERT INTO auth_sessions(token, user_id, expires_at, active)
        VALUES(?, ?, ?, 1)
        """,
        (token, user_id, expires_at),
      )
      conn.commit()

  return {
    "ok": True,
    "token": token,
    "expires_at": expires_at,
    "user": {"id": user_id, "name": payload.name.strip(), "email": email},
  }


@app.post("/auth/login")
def auth_login(payload: LoginPayload):
  email = _norm_email(payload.email)
  with _LOCK:
    with _get_conn() as conn:
      row = conn.execute(
        "SELECT id, name, email, password_hash, password_salt, active FROM users WHERE email = ?",
        (email,),
      ).fetchone()
      if row is None or int(row["active"]) != 1:
        return {"ok": False, "reason": "invalid-credentials"}

      expected_hash = str(row["password_hash"])
      candidate_hash = _hash_password(payload.password, str(row["password_salt"]))
      if not hmac.compare_digest(expected_hash, candidate_hash):
        return {"ok": False, "reason": "invalid-credentials"}

      conn.execute("UPDATE auth_sessions SET active = 0 WHERE user_id = ?", (int(row["id"]),))
      token = _new_session_token()
      expires_at = _session_expiration_iso()
      conn.execute(
        """
        INSERT INTO auth_sessions(token, user_id, expires_at, active)
        VALUES(?, ?, ?, 1)
        """,
        (token, int(row["id"]), expires_at),
      )
      conn.commit()

  return {
    "ok": True,
    "token": token,
    "expires_at": expires_at,
    "user": {"id": int(row["id"]), "name": str(row["name"]), "email": str(row["email"])}
  }


@app.get("/auth/me")
def auth_me(token: str):
  with _get_conn() as conn:
    row = _auth_user_from_token(conn, token)
    if row is None:
      return {"ok": False, "reason": "invalid-session"}
  return {
    "ok": True,
    "user": {"id": int(row["id"]), "name": str(row["name"]), "email": str(row["email"])},
    "expires_at": str(row["expires_at"]),
  }


@app.post("/auth/logout")
def auth_logout(payload: LogoutPayload):
  with _LOCK:
    with _get_conn() as conn:
      conn.execute("UPDATE auth_sessions SET active = 0 WHERE token = ?", (payload.token,))
      conn.commit()
  return {"ok": True}


@app.post("/risk/score")
async def risk_score(payload: RiskPayload) -> dict[str, Any]:
  with _LOCK:
    with _get_conn() as conn:
      existing_count = conn.execute(
        "SELECT COUNT(*) AS cnt FROM captures_local WHERE device_id = ?",
        (payload.device_id,),
      ).fetchone()["cnt"]

      global_prob = _global_model_probability(payload.metrics)
      if global_prob is None:
        global_prob = _heuristic_probability(payload.metrics)

      personal_prob = None
      if int(existing_count) >= PERSONAL_TRAIN_INTERVAL:
        personal_prob = _personal_model_probability(payload.device_id, payload.metrics)

      if personal_prob is not None:
        source = "fog-personal+global"
        final_prob = 0.55 * float(personal_prob) + 0.25 * float(global_prob) + 0.20 * payload.local_probability
      else:
        source = "fog-global"
        final_prob = 0.6 * float(global_prob) + 0.4 * payload.local_probability

      final_prob = max(0.0, min(1.0, final_prob))

      rmssd = float(payload.metrics.get("rmssd", 0.0))
      sdnn = float(payload.metrics.get("sdnn", 0.0))
      pnn50 = float(payload.metrics.get("pnn50", 0.0))
      mean_hr = float(payload.metrics.get("mean_hr", 0.0))

      conn.execute(
        """
        INSERT INTO captures_local(
          device_id, user_id, sampling_hz, rmssd, sdnn, pnn50, mean_hr,
          local_probability, final_probability, high_stress
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          payload.device_id,
          payload.user_id,
          payload.sampling_hz,
          rmssd,
          sdnn,
          pnn50,
          mean_hr,
          payload.local_probability,
          final_prob,
          1 if final_prob >= 0.70 else 0,
        ),
      )

      sample_count = conn.execute(
        "SELECT COUNT(*) AS cnt FROM captures_local WHERE device_id = ?",
        (payload.device_id,),
      ).fetchone()["cnt"]
      model_version = _get_model_version(conn)

      conn.execute(
        """
        INSERT INTO device_model_state(device_id, samples_seen, last_personal_version, last_global_version, updated_at)
        VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(device_id) DO UPDATE SET
          samples_seen=excluded.samples_seen,
          updated_at=CURRENT_TIMESTAMP
        """,
        (payload.device_id, int(sample_count), model_version, model_version),
      )
      conn.commit()

      personal_train = {"trained": False, "reason": "not-due"}
      if int(sample_count) >= PERSONAL_TRAIN_INTERVAL and int(sample_count) % PERSONAL_TRAIN_INTERVAL == 0:
        personal_train = _train_personal_model(conn, payload.device_id, int(sample_count))
        conn.commit()

      train_result = _try_local_federated_round(conn)
      model_version = _get_model_version(conn)

  return {
    "user_id": payload.user_id,
    "device_id": payload.device_id,
    "risk_probability": final_prob,
    "source": source,
    "model_version": int(model_version),
    "personal_model": personal_train,
    "train_result": train_result,
  }


@app.get("/federated/local-status")
def federated_local_status():
  with _get_conn() as conn:
    model_version = _get_model_version(conn)
    rounds = conn.execute("SELECT COUNT(*) AS cnt FROM local_rounds").fetchone()["cnt"]
    samples_total = conn.execute("SELECT COUNT(*) AS cnt FROM captures_local").fetchone()["cnt"]
    pending = conn.execute("SELECT COUNT(*) AS cnt FROM captures_local WHERE trained_round IS NULL").fetchone()["cnt"]
    devices = conn.execute("SELECT COUNT(*) AS cnt FROM device_model_state").fetchone()["cnt"]
    personal_rounds = conn.execute("SELECT COUNT(*) AS cnt FROM personal_rounds").fetchone()["cnt"]
    per_device = [
      dict(row)
      for row in conn.execute(
        "SELECT device_id, samples_seen, last_personal_version, last_global_version, updated_at FROM device_model_state ORDER BY updated_at DESC"
      ).fetchall()
    ]

  return {
    "fog_id": FOG_ID,
    "model_version": int(model_version),
    "rounds": int(rounds),
    "samples_total": int(samples_total),
    "pending_samples": int(pending),
    "devices": int(devices),
    "personal_rounds": int(personal_rounds),
    "device_models": per_device,
    "personal_train_interval": PERSONAL_TRAIN_INTERVAL,
    "db_path": str(DB_PATH),
  }


@app.get("/federated/training-metrics")
def federated_training_metrics(limit: int = Query(default=20, ge=1, le=500)):
  with _get_conn() as conn:
    global_rows = [
      dict(row)
      for row in conn.execute(
        """
        SELECT round_version, created_at, clients, samples, mae, rmse, smape, mase
        FROM local_rounds
        ORDER BY round_version DESC
        LIMIT ?
        """,
        (limit,),
      ).fetchall()
    ]
    personal_rows = [
      dict(row)
      for row in conn.execute(
        """
        SELECT id, created_at, device_id, personal_version, samples, mae, rmse
        FROM personal_rounds
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit * 5,),
      ).fetchall()
    ]

  latest_global = global_rows[0] if global_rows else None
  latest_personal_by_device: dict[str, dict[str, Any]] = {}
  for row in personal_rows:
    device_id = str(row["device_id"])
    if device_id not in latest_personal_by_device:
      latest_personal_by_device[device_id] = row

  return {
    "global_model": {
      "latest": latest_global,
      "history": global_rows,
      "rounds": len(global_rows),
    },
    "personal_models": {
      "latest_by_device": list(latest_personal_by_device.values()),
      "history": personal_rows,
      "trained_devices": len(latest_personal_by_device),
    },
  }


@app.post("/federated/simulate-profiles")
def federated_simulate_profiles(payload: SimulateProfilesPayload):
  inserted = 0
  rng = random.Random(int(payload.seed))
  personal_results: list[dict[str, Any]] = []
  global_rounds: list[dict[str, Any]] = []

  with _LOCK:
    with _get_conn() as conn:
      for profile_key, spec in PROFILE_SPECS.items():
        for _ in range(int(payload.samples_per_profile)):
          metrics = _sample_profile_metrics(spec, rng)
          local_prob = _heuristic_probability(metrics)
          noise = rng.uniform(-0.05, 0.05)
          final_prob = max(
            0.0,
            min(
              1.0,
              0.72 * float(local_prob) + 0.28 * float(spec["stress_bias"]) + noise,
            ),
          )
          conn.execute(
            """
            INSERT INTO captures_local(
              device_id, user_id, sampling_hz, rmssd, sdnn, pnn50, mean_hr,
              local_probability, final_probability, high_stress
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
              spec["device_id"],
              spec["user_id"],
              float(payload.sampling_hz),
              float(metrics["rmssd"]),
              float(metrics["sdnn"]),
              float(metrics["pnn50"]),
              float(metrics["mean_hr"]),
              float(local_prob),
              float(final_prob),
              1 if float(final_prob) >= 0.70 else 0,
            ),
          )
          inserted += 1

        sample_count = conn.execute(
          "SELECT COUNT(*) AS cnt FROM captures_local WHERE device_id = ?",
          (spec["device_id"],),
        ).fetchone()["cnt"]
        model_version = _get_model_version(conn)
        conn.execute(
          """
          INSERT INTO device_model_state(device_id, samples_seen, last_personal_version, last_global_version, updated_at)
          VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
          ON CONFLICT(device_id) DO UPDATE SET
            samples_seen=excluded.samples_seen,
            last_global_version=excluded.last_global_version,
            updated_at=CURRENT_TIMESTAMP
          """,
          (spec["device_id"], int(sample_count), model_version, model_version),
        )

      conn.commit()

      for spec in PROFILE_SPECS.values():
        sample_count = conn.execute(
          "SELECT COUNT(*) AS cnt FROM captures_local WHERE device_id = ?",
          (spec["device_id"],),
        ).fetchone()["cnt"]
        personal_results.append(_train_personal_model(conn, spec["device_id"], int(sample_count)))
      conn.commit()

      if payload.run_global_training:
        while True:
          result = _try_local_federated_round(conn)
          if not result.get("trained"):
            break
          global_rounds.append(result)

      metrics_snapshot = federated_training_metrics(limit=20)

  return {
    "ok": True,
    "inserted_samples": int(inserted),
    "profiles": len(PROFILE_SPECS),
    "samples_per_profile": int(payload.samples_per_profile),
    "personal_training": personal_results,
    "global_rounds_triggered": global_rounds,
    "metrics": metrics_snapshot,
  }


@app.get("/federated/dashboard", response_class=HTMLResponse)
def federated_dashboard() -> str:
  return """
<!doctype html>
<html lang="pt-br" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Olivia Federated Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            slate: { 850: '#151e32', 900: '#0f172a', 950: '#020617' },
            primary: { 400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed' },
            emerald: { 400: '#34d399', 500: '#10b981' },
          },
          fontFamily: { sans: ['Inter', 'sans-serif'] }
        }
      }
    }
  </script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .glass-panel {
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
      transition: all 0.3s ease;
    }
    .glass-panel:hover {
      background: rgba(30, 41, 59, 0.9);
      border-color: rgba(139, 92, 246, 0.3);
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    }
    .status-pill {
      background: rgba(16, 185, 129, 0.1);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .table-head {
      background: rgba(15, 23, 42, 0.8);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      color: #94a3b8;
      font-weight: 600;
      border-bottom: 2px solid #1e293b;
    }
    .orbit-bg {
      position: fixed;
      width: 600px;
      height: 600px;
      background: radial-gradient(circle, rgba(139, 92, 246, 0.08) 0%, rgba(15, 23, 42, 0) 70%);
      top: -200px;
      left: -200px;
      filter: blur(80px);
      z-index: -1;
      animation: float 20s infinite ease-in-out;
    }
    @keyframes float { 0% { transform: translate(0,0); } 50% { transform: translate(50px, 50px); } 100% { transform: translate(0,0); } }
  </style>
</head>
<body class="bg-slate-950 text-slate-200 min-h-screen flex selection:bg-violet-500 selection:text-white overflow-x-hidden">

  <!-- Background Ambience -->
  <div class="orbit-bg"></div>
  <div class="orbit-bg" style="top:auto; bottom:-200px; left:auto; right:-200px; background: radial-gradient(circle, rgba(16, 185, 129, 0.05) 0%, rgba(15, 23, 42, 0) 70%); animation-delay: -5s;"></div>

  <!-- Sidebar -->
  <aside class="w-64 bg-slate-900 border-r border-slate-800 flex-col hidden md:flex fixed h-full z-20">
    <div class="p-6 border-b border-slate-800">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center font-bold text-white shadow-lg shadow-violet-500/20">O</div>
        <span class="text-xl font-bold tracking-tight text-white">Olivia<span class="text-violet-500">.ai</span></span>
      </div>
    </div>
    
    <nav class="flex-1 overflow-y-auto py-6 px-3 space-y-1">
      <div class="px-3 mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">Platform</div>
      <a href="/control/dashboard" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 cursor-pointer hover:bg-slate-800 hover:text-white transition-colors">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
        Control Center
      </a>
      <a href="/federated/dashboard" class="sidebar-link active flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium bg-violet-900/20 text-violet-300 border border-violet-800/30">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
        Federated Status
      </a>
      <a href="/evaluation/dashboard" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 cursor-pointer hover:bg-slate-800 hover:text-white transition-colors">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
        Evaluation
      </a>
    </nav>
  </aside>

  <!-- Main Content -->
  <main class="flex-1 md:ml-64 p-8">
    <header class="flex items-center justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold text-white tracking-tight">Federated Learning Monitor</h1>
        <p class="text-slate-400 mt-1">Real-time status of local training and device synchronization.</p>
      </div>
      <div class="flex items-center gap-2">
         <div class="flex items-center space-x-2 bg-slate-900 px-4 py-2 rounded-full border border-slate-800 shadow-sm">
            <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            <span class="text-xs font-bold text-emerald-400 uppercase tracking-wider">System Online</span>
         </div>
      </div>
    </header>

    <!-- KPI Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8" id="kpi-grid">
       <!-- Loading KPIs -->
       <div class="col-span-4 h-32 glass-panel rounded-xl animate-pulse"></div>
    </div>

    <!-- Devices Table -->
    <div class="glass-panel rounded-xl overflow-hidden shadow-lg shadow-black/20">
       <div class="px-6 py-5 border-b border-slate-700/50 flex items-center justify-between bg-slate-900/30">
          <h3 class="text-lg font-semibold text-slate-200">Connected Devices</h3>
          <button onclick="load()" class="p-2 hover:bg-slate-700 rounded-lg transition-colors">
             <svg class="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
          </button>
       </div>
       <div class="overflow-x-auto">
          <table class="w-full text-sm text-left">
             <thead class="table-head">
               <tr>
                 <th class="px-6 py-4">Device ID</th>
                 <th class="px-6 py-4">Training Progress</th>
                 <th class="px-6 py-4">Total Samples</th>
                 <th class="px-6 py-4">Model Ver.</th>
                 <th class="px-6 py-4 text-right">Last Sync</th>
               </tr>
             </thead>
             <tbody id="rows" class="divide-y divide-slate-800/50 text-slate-300">
                <!-- Rows injected here -->
             </tbody>
          </table>
       </div>
    </div>
  </main>

  <script>
    const fmtTime = (ts) => {
       if(!ts) return '-';
       const d = new Date(ts.replace(' ', 'T'));
       return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    };

    const kpiCard = (title, value, icon, gradient) => `
       <div class="glass-panel p-6 rounded-xl relative overflow-hidden group hover:border-violet-500/30 transition-all duration-300">
          <div class="absolute right-0 top-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity text-white">
             ${icon}
          </div>
          <p class="text-sm font-medium text-slate-400 uppercase tracking-widest mb-1">${title}</p>
          <div class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r ${gradient}">${value}</div>
       </div>
    `;

    async function load(){
      try {
        const res = await fetch('/federated/local-status');
        const data = await res.json();
        
        // Render KPIs
        const kpis = document.getElementById('kpi-grid');
        kpis.innerHTML = 
          kpiCard('Model Version', `v${data.model_version}.0`, 
             '<svg class="w-16 h-16" fill="currentColor" viewBox="0 0 20 20"><path d="M11 17a1 1 0 001.447.894l4-2A1 1 0 0017 15V9.236a1 1 0 00-1.447-.894l-4 2a1 1 0 00-.553.894V17zM15.211 6.276a1 1 0 000-1.788l-4.764-2.382a1 1 0 00-.894 0L4.789 4.488a1 1 0 000 1.788l4.764 2.382a1 1 0 00.894 0l4.764-2.382zM4.447 8.342A1 1 0 003 9.236V15a1 1 0 00.553.894l4 2A1 1 0 009 17v-5.764a1 1 0 00-.553-.894l-4-2z"/></svg>', 
             'from-violet-400 to-fuchsia-400') +
          kpiCard('Global Rounds', data.rounds, 
             '<svg class="w-16 h-16" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/></svg>', 
             'from-blue-400 to-cyan-400') +
          kpiCard('Total Samples', data.samples_total, 
             '<svg class="w-16 h-16" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M3 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-.293.707L12 11.414V15a1 1 0 01-.293.707l-2 2A1 1 0 018 17v-5.586L3.293 6.707A1 1 0 013 6V3z" clip-rule="evenodd"/></svg>', 
             'from-emerald-400 to-teal-400') +
          kpiCard('Active Devices', data.devices, 
             '<svg class="w-16 h-16" fill="currentColor" viewBox="0 0 20 20"><path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z"/></svg>', 
             'from-orange-400 to-amber-400');

        // Render Table
        const tbody = document.getElementById('rows');
        tbody.innerHTML = '';
        (data.device_models || []).forEach(d => {
          const nextAt = ((Math.floor((d.samples_seen||0)/data.personal_train_interval)+1)*data.personal_train_interval);
          const progress = Math.min(100, (d.samples_seen % data.personal_train_interval) / data.personal_train_interval * 100);
          
          const tr = document.createElement('tr');
          tr.className = "hover:bg-slate-800/50 transition-colors";
          tr.innerHTML = `
            <td class="px-6 py-4 font-medium text-white flex items-center gap-3">
               <div class="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-xs text-slate-500 font-mono border border-slate-700">D</div>
               ${d.device_id}
            </td>
            <td class="px-6 py-4">
              <div class="flex items-center gap-3">
                 <div class="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div class="h-full bg-violet-500 rounded-full" style="width: ${progress}%"></div>
                 </div>
                 <span class="text-xs text-slate-400 w-12 text-right">${d.samples_seen}/${nextAt}</span>
              </div>
            </td>
            <td class="px-6 py-4 font-mono text-slate-400">${d.samples_seen}</td>
            <td class="px-6 py-4">
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-800/80 text-violet-300 border border-violet-500/20">
                v${d.last_personal_version}
              </span>
            </td>
            <td class="px-6 py-4 text-right text-slate-500 font-mono text-xs">
              ${d.updated_at}
            </td>`;
          tbody.appendChild(tr);
        });
      } catch(e) { console.error(e); }
    }
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
  """


@app.get("/evaluation/summary")
def evaluation_summary(force: bool = Query(default=False)):
  return _compute_eval_summary(force=force)


@app.get("/evaluation/series")
def evaluation_series(device_id: str, horizon: int = Query(default=1, ge=1, le=30)):
  with _get_conn() as conn:
    rows = _device_rows(conn, device_id)
  out = _evaluate_device_horizon(rows, horizon, device_id)
  if out is None:
    return {
      "ok": False,
      "message": "Dados insuficientes para avaliação (mínimo recomendado: ~40 amostras por device).",
    }

  return {
    "ok": True,
    "device_id": device_id,
    "horizon": horizon,
    "model_source": out.get("model_source"),
    "selected_predictor": out.get("selected_predictor"),
    "baseline_naive": out.get("baseline_naive"),
    "metrics": out["metrics"],
    "error_by_horizon": {
      "horizon": f"t+{horizon}",
      "mae": out["metrics"]["mae"],
      "rmse": out["metrics"]["rmse"],
    },
    "series": out["series"],
    "backtesting": _backtesting_windows(out["series"]["actual"], out["series"]["forecast"]),
    "interval": out["interval"],
  }


@app.get("/evaluation/dashboard", response_class=HTMLResponse)
def evaluation_dashboard() -> str:
  return """
<!doctype html>
<html lang="pt-br" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Olivia Evaluation PRO</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            slate: { 850: '#151e32', 900: '#0f172a', 950: '#020617' },
            primary: { 400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed' },
          },
          fontFamily: { sans: ['Inter', 'sans-serif'] }
        }
      }
    }
  </script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .glass-panel {
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #0f172a; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    .sidebar-link { transition: all 0.2s; border-left: 3px solid transparent; }
    .sidebar-link:hover, .sidebar-link.active {
      background: linear-gradient(90deg, rgba(139,92,246,0.1), transparent);
      border-left-color: #8b5cf6;
      color: #fff;
    }
    canvas { min-height: 250px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-200 h-screen overflow-hidden flex selection:bg-violet-500 selection:text-white">

  <!-- Sidebar -->
  <aside class="w-64 bg-slate-900 border-r border-slate-800 flex flex-col hidden md:flex z-20">
    <div class="p-6 border-b border-slate-800">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center font-bold text-white shadow-lg shadow-violet-500/20">O</div>
        <span class="text-xl font-bold tracking-tight text-white">Olivia<span class="text-violet-500">.ai</span></span>
      </div>
    </div>
    
    <nav class="flex-1 overflow-y-auto py-6 px-3 space-y-1">
      <div class="px-3 mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">Analytics</div>
      <a href="/control/dashboard" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 cursor-pointer hover:bg-slate-800 hover:text-white transition-colors">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
        Control Center
      </a>
      <a href="#" class="sidebar-link active flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-200">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
        Evaluation (Deep Dive)
      </a>
    </nav>
  </aside>

  <!-- Main Content -->
  <main class="flex-1 flex flex-col h-full overflow-hidden relative">
    <header class="h-16 border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-md flex items-center justify-between px-6 z-10">
      <div class="flex items-center gap-4">
        <h2 class="text-lg font-semibold text-white">Advanced Evaluation</h2>
        <div class="h-6 w-px bg-slate-700 mx-2"></div>
        <div class="flex items-center gap-2">
           <select id="deviceSelect" class="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded px-3 py-1 outline-none min-w-[200px]"></select>
           <select id="horizonSelect" class="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded px-3 py-1 outline-none">
             <option value="1">t+1</option>
             <option value="7">t+7</option>
             <option value="30">t+30</option>
           </select>
           <button id="refreshBtn" class="px-3 py-1 bg-violet-600 hover:bg-violet-700 rounded text-xs font-bold text-white uppercase tracking-wider transition-colors ml-2">Analyze</button>
        </div>
      </div>
    </header>

    <div class="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-6">
      <!-- KPIs -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4" id="kpis">
         <!-- Injected via JS -->
      </div>
      
      <!-- Charts Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="glass-panel p-5 rounded-xl border border-slate-700/40 lg:col-span-2">
          <h3 class="text-lg font-medium text-slate-200 mb-4">Forecast vs Real Signal</h3>
          <div class="h-[300px] w-full relative">
            <canvas id="lineChart"></canvas>
          </div>
        </div>

        <div class="glass-panel p-5 rounded-xl border border-slate-700/40">
           <h3 class="text-lg font-medium text-slate-200 mb-4">Error Distribution by Horizon</h3>
           <div class="h-[250px] w-full relative">
             <canvas id="barChart"></canvas>
           </div>
        </div>

        <div class="glass-panel p-5 rounded-xl border border-slate-700/40">
           <h3 class="text-lg font-medium text-slate-200 mb-4">Backtesting Consistency (Wins)</h3>
           <div class="h-[250px] w-full relative">
             <canvas id="backtestChart"></canvas>
           </div>
        </div>
        
        <div class="glass-panel p-5 rounded-xl border border-slate-700/40">
           <h3 class="text-lg font-medium text-white mb-4">Uncertainty Intervals</h3>
           <div class="h-[250px] w-full relative">
             <canvas id="intervalChart"></canvas>
           </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';
    Chart.defaults.font.family = 'Inter';
    
    let lineChart, barChart, backtestChart, intervalChart;
    const fmt = (v)=> (v===null || v===undefined || Number.isNaN(v)) ? 'n/a' : Number(v).toFixed(4);

    async function loadSummary(){
      const res = await fetch('/evaluation/summary');
      return await res.json();
    }

    async function loadSeries(deviceId,horizon){
      const res = await fetch(`/evaluation/series?device_id=${encodeURIComponent(deviceId)}&horizon=${horizon}`);
      return await res.json();
    }

    const kpiEl = (lbl, val, sub="") => `
       <div class="glass-panel p-4 rounded-xl flex flex-col justify-between h-full">
          <span class="text-xs uppercase text-slate-400 font-bold tracking-wider">${lbl}</span>
          <div class="text-2xl font-bold text-white mt-1">${val}</div>
          <div class="text-xs text-slate-500 mt-1">${sub}</div>
       </div>
    `;

    function renderKPIs(summary, seriesData){
      const k = summary.global_overall;
      const el = document.getElementById('kpis');
      
      let html = kpiEl('Devices', summary.devices_with_eval, 'Total evaluatable');
      
      if(seriesData && seriesData.ok){
         // Specific
         const m = seriesData.metrics;
         html += kpiEl('Current MAE', fmt(m.mae), `Baseline: ${fmt((seriesData.baseline_naive||{}).mae)}`);
         html += kpiEl('Current RMSE', fmt(m.rmse), 'Loss metric');
         html += kpiEl('sMAPE', fmt(m.smape), 'Percentage error');
         html += kpiEl('Model Source', seriesData.model_source?.substring(0,10), `Pred: ${seriesData.selected_predictor}`);
      } else {
         // Global
         html += kpiEl('Global MAE', fmt(k.mae), 'Average across all');
         html += kpiEl('Global RMSE', fmt(k.rmse), 'Average across all');
         html += kpiEl('Global sMAPE', fmt(k.smape), 'Average across all');
      }
      el.innerHTML = html;
    }

    function ensureChart(chartRef, ctx, config){
      if(chartRef){ chartRef.destroy(); }
      return new Chart(ctx, config);
    }

    async function render(){
      const summary = await loadSummary();
      const sel = document.getElementById('deviceSelect');
      
      // Fill select if empty
      if(sel.options.length===0){
        summary.devices.forEach(d=>{
          const o = document.createElement('option');
          o.value = d.device_id;
          o.textContent = `${d.device_id}`;
          sel.appendChild(o);
        });
      }
      
      if(summary.devices.length===0){ 
         renderKPIs(summary, null);
         return; 
      }

      const deviceId = sel.value || summary.devices[0].device_id;
      if(!sel.value) sel.value = deviceId;
      
      const horizon = Number(document.getElementById('horizonSelect').value);
      const series = await loadSeries(deviceId, horizon);
      
      renderKPIs(summary, series);
      if(!series.ok){ return; }

      const ts = series.series.timestamps;
      const actual = series.series.actual;
      const forecast = series.series.forecast;
      const lower = series.series.lower;
      const upper = series.series.upper;

      const ctxLine = document.getElementById('lineChart').getContext('2d');
      const gradLine = ctxLine.createLinearGradient(0,0,0,400);
      gradLine.addColorStop(0, 'rgba(139, 92, 246, 0.2)');
      gradLine.addColorStop(1, 'rgba(139, 92, 246, 0)');

      lineChart = ensureChart(lineChart, ctxLine, {
        type:'line',
        data:{labels:ts,datasets:[
          {label:'Real Signal',data:actual,borderColor:'#10b981',pointRadius:0,borderWidth:2, tension:0.1},
          {label:'Forecast',data:forecast,borderColor:'#8b5cf6',backgroundColor:gradLine, fill:true, pointRadius:0,borderWidth:2, tension:0.4},
          {label:'Lower Bound',data:lower,borderColor:'#f59e0b',pointRadius:0,borderDash:[4,4], borderWidth:1, hidden:true},
          {label:'Upper Bound',data:upper,borderColor:'#f59e0b',pointRadius:0,borderDash:[4,4], borderWidth:1, hidden:true}
        ]},
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: { legend: { position: 'top', labels: {color: '#94a3b8'} } },
          scales: { x: { display: false }, y: { grid: { color: '#334155' }, ticks: {color: '#94a3b8'} } }
        }
      });

      const dev = summary.devices.find(d=>d.device_id===deviceId);
      const labels = Object.keys(dev.horizons || {});
      const maes = labels.map(h=>dev.horizons[h].mae);
      const rmses = labels.map(h=>dev.horizons[h].rmse);
      
      barChart = ensureChart(barChart, document.getElementById('barChart'), {
        type:'bar',
        data:{labels,datasets:[
          {label:'MAE',data:maes,backgroundColor:'#38bdf8', borderRadius: 4},
          {label:'RMSE',data:rmses,backgroundColor:'#818cf8', borderRadius: 4}
        ]},
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: '#334155' }, ticks: {color: '#94a3b8'} }, x: {ticks: {color: '#94a3b8'}} } }
      });

      const bt = series.backtesting || [];
      backtestChart = ensureChart(backtestChart, document.getElementById('backtestChart'), {
        type:'line',
        data:{labels:bt.map(x=>`W${x.window}`),datasets:[
          {label:'MAE',data:bt.map(x=>x.mae),borderColor:'#34d399', tension: 0.3},
          {label:'RMSE',data:bt.map(x=>x.rmse),borderColor:'#fb7185', tension: 0.3}
        ]},
        options: { responsive: true, maintainAspectRatio: false, scales: { x: {display: false}, y: { grid: { color: '#334155' }, ticks: {color: '#94a3b8'} }} }
      });

      intervalChart = ensureChart(intervalChart, document.getElementById('intervalChart'), {
        type:'bar',
        data:{labels:['Coverage','Avg Width'],datasets:[
          {label:'Metrics',data:[series.interval.coverage, series.interval.avg_width],backgroundColor:['#fbbf24','#f97316'], borderRadius: 6}
        ]},
        options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y', scales: { x: { grid: { color: '#334155' }, ticks: {color: '#94a3b8'} }, y: {ticks: {color: '#94a3b8'}} } }
      });
    }

    document.getElementById('refreshBtn').addEventListener('click',render);
    document.getElementById('deviceSelect').addEventListener('change',render);
    document.getElementById('horizonSelect').addEventListener('change',render);
    render();
  </script>
</body>
</html>
  """


@app.get("/control/dashboard", response_class=HTMLResponse)
def control_dashboard() -> str:
  return """
<!doctype html>
<html lang="pt-br" class="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Olivia Control Center - Premium</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            slate: { 850: '#151e32', 900: '#0f172a', 950: '#020617' },
            primary: { 400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed' },
            emerald: { 400: '#34d399', 500: '#10b981' },
            rose: { 400: '#fb7185', 500: '#f43f5e' }
          },
          fontFamily: { sans: ['Inter', 'sans-serif'] }
        }
      }
    }
  </script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    /* Glassmorphism subtle */
    .glass-panel {
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .gradient-text {
      background: linear-gradient(to right, #a78bfa, #2dd4bf);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #0f172a; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    .card-hover:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
      border-color: rgba(139, 92, 246, 0.3);
    }
    .sidebar-link { transition: all 0.2s; border-left: 3px solid transparent; }
    .sidebar-link:hover, .sidebar-link.active {
      background: linear-gradient(90deg, rgba(139,92,246,0.1), transparent);
      border-left-color: #8b5cf6;
      color: #fff;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-200 h-screen overflow-hidden flex selection:bg-violet-500 selection:text-white">

  <!-- Sidebar -->
  <aside class="w-64 bg-slate-900 border-r border-slate-800 flex flex-col hidden md:flex z-20">
    <div class="p-6 border-b border-slate-800/50">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center font-bold text-white shadow-lg shadow-violet-500/20">O</div>
        <span class="text-xl font-bold tracking-tight text-white">Olivia<span class="text-violet-400">.ai</span></span>
      </div>
    </div>
    
    <nav class="flex-1 overflow-y-auto py-6 px-3 space-y-1">
      <div class="px-3 mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">Main Menu</div>
      <a href="/control/dashboard" class="sidebar-link active flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-300">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
        Dashboard
      </a>
      <a href="/evaluation/dashboard" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 cursor-pointer hover:bg-slate-800 hover:text-white transition-colors">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
        Evaluation (Detail)
      </a>
      <a href="/federated/dashboard" class="sidebar-link flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 cursor-pointer hover:bg-slate-800 hover:text-white transition-colors">
        <svg class="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
        Federated Nodes
      </a>
    </nav>
    
    <div class="p-4 border-t border-slate-800">
       <div class="flex items-center gap-3">
         <div class="w-8 h-8 rounded-full bg-slate-700 border border-slate-600"></div>
         <div>
           <div class="text-sm font-medium text-white">Fog Admin</div>
           <div class="text-xs text-slate-500">Active</div>
         </div>
       </div>
    </div>
  </aside>

  <!-- Main Content -->
  <main class="flex-1 flex flex-col h-full overflow-hidden relative">
    <!-- Top Bar -->
    <header class="h-16 border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-6 z-10">
      <h2 class="text-lg font-semibold text-white">Fog Control Center</h2>
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 rounded-full border border-slate-700/50">
           <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
           <span class="text-xs font-medium text-emerald-400">System Online</span>
        </div>
        <button id="refreshBtn" class="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg shadow-lg shadow-violet-600/20 transition-all active:scale-95">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
          Refresh Data
        </button>
      </div>
    </header>

    <div class="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-6">
      
      <!-- Actions Row -->
      <div class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
         <div class="col-span-1 md:col-span-2 lg:col-span-3 glass-panel rounded-xl p-5 relative overflow-hidden group">
            <div class="absolute top-0 right-0 w-64 h-64 bg-violet-600/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 group-hover:bg-violet-600/20 transition-all duration-700"></div>
            <h3 class="text-lg font-semibold text-white mb-2">Simulação & Treino</h3>
            <p class="text-sm text-slate-400 mb-4 max-w-2xl">Execute pipeline completo: Geração de dados sintéticos (5 perfis) -> Treino Local -> Agregação -> Treino Global.</p>
            <div class="flex flex-wrap gap-3">
               <button id="simulateBtn" class="px-4 py-2 bg-white text-slate-900 hover:bg-slate-200 font-semibold text-sm rounded-lg shadow-lg transition-colors flex items-center gap-2">
                 <svg class="w-4 h-4 text-violet-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                 Run Full Simulation Pipeline
               </button>
               <a href="/federated/dashboard" class="px-4 py-2 border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white font-medium text-sm rounded-lg transition-colors inline-block no-underline">
                 Check Cloud Coordinator
               </a>
            </div>
         </div>
         <div class="glass-panel rounded-xl p-5 flex flex-col justify-between">
           <div class="text-slate-400 text-xs font-semibold uppercase tracking-wider">Status Atual</div>
           <div id="statusBadges" class="flex flex-col gap-2 mt-2">
              <div class="h-6 w-24 bg-slate-800 rounded animate-pulse"></div>
              <div class="h-6 w-32 bg-slate-800 rounded animate-pulse"></div>
           </div>
         </div>
      </div>

      <!-- Stats Grid -->
      <div id="kpis" class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <!-- Filled by JS -->
        <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 items-start justify-center min-h-[100px] animate-pulse"></div>
        <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 items-start justify-center min-h-[100px] animate-pulse"></div>
        <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 items-start justify-center min-h-[100px] animate-pulse"></div>
        <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 items-start justify-center min-h-[100px] animate-pulse"></div>
      </div>

      <!-- Main Columns -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Personal Models -->
        <div class="glass-panel rounded-xl border border-slate-700/50 flex flex-col">
          <div class="p-5 border-b border-slate-800 flex justify-between items-center">
             <h3 class="font-semibold text-white flex items-center gap-2">
               <svg class="w-5 h-5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
               Modelos Pessoais <span class="text-xs font-normal text-slate-500 ml-2">(Latest Snapshot)</span>
             </h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm text-left">
              <thead class="text-xs text-slate-400 uppercase bg-slate-900/50">
                <tr>
                  <th class="px-5 py-3">Device</th>
                  <th class="px-5 py-3">Ver</th>
                  <th class="px-5 py-3">Samples</th>
                  <th class="px-5 py-3 text-right">MAE</th>
                  <th class="px-5 py-3 text-right">RMSE</th>
                </tr>
              </thead>
              <tbody id="personalRows" class="divide-y divide-slate-800">
                <!-- Rows -->
              </tbody>
            </table>
          </div>
        </div>

        <!-- Global History -->
        <div class="glass-panel rounded-xl border border-slate-700/50 flex flex-col">
           <div class="p-5 border-b border-slate-800 flex justify-between items-center">
             <h3 class="font-semibold text-white flex items-center gap-2">
               <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
               Modelo Global <span class="text-xs font-normal text-slate-500 ml-2">(History)</span>
             </h3>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm text-left">
              <thead class="text-xs text-slate-400 uppercase bg-slate-900/50">
                <tr>
                  <th class="px-5 py-3">Round</th>
                  <th class="px-5 py-3">Clients</th>
                  <th class="px-5 py-3">Samples</th>
                  <th class="px-5 py-3 text-right">MAE</th>
                  <th class="px-5 py-3 text-right">RMSE</th>
                </tr>
              </thead>
              <tbody id="globalRows" class="divide-y divide-slate-800">
                <!-- Rows -->
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Quick Eval -->
      <div class="glass-panel rounded-xl p-5 border border-slate-700/50">
         <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
            <div>
              <h3 class="text-lg font-semibold text-white">Avaliação Rápida</h3>
              <p class="text-sm text-slate-400">Verifique a performance de predição para um usuário específico.</p>
            </div>
            <div class="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-lg border border-slate-700">
               <select id="deviceSelect" class="bg-transparent border-none text-sm text-slate-200 focus:ring-0 cursor-pointer min-w-[140px] appearance-none px-3 font-medium outline-none">
                 <option>Carregando...</option>
               </select>
               <div class="w-px h-4 bg-slate-700"></div>
               <select id="horizonSelect" class="bg-transparent border-none text-sm text-slate-200 focus:ring-0 cursor-pointer appearance-none px-3 font-medium outline-none">
                 <option value="1">Horizonte: t+1</option>
                 <option value="7">Horizonte: t+7</option>
                 <option value="30">Horizonte: t+30</option>
               </select>
               <button id="loadEvalBtn" class="bg-violet-600 hover:bg-violet-500 text-white rounded px-3 py-1.5 text-xs font-bold uppercase transition-colors">
                 Go
               </button>
            </div>
         </div>
         
         <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-1 space-y-3" id="evalKpisInput">
               <div class="p-3 rounded-lg border border-dashed border-slate-700 text-slate-500 text-sm text-center">Selecione um device e clique em GO</div>
            </div>
            <div class="lg:col-span-2 bg-slate-900/50 rounded-xl p-4 border border-slate-800/50 relative min-h-[200px]">
               <canvas id="evalChart"></canvas>
            </div>
         </div>
      </div>
      
      <div class="text-center text-xs text-slate-600 pb-4">Olivia Stress Platform &copy; 2026. Powered by Fog Orchestrator.</div>

    </div>
  </main>

  <script>
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';
    Chart.defaults.font.family = 'Inter';
    let evalChart;
    const fmt = (v) => (v===null || v===undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(4);

    async function getJson(url, opts){
      const res = await fetch(url, opts);
      if(!res.ok){ throw new Error(`HTTP ${res.status}`); }
      return await res.json();
    }
    
    // UI Helpers
    const kpiCard = (title, value, sub, colorClass='text-white') => `
      <div class="glass-panel p-4 rounded-xl flex flex-col gap-1 card-hover transition-all">
         <span class="text-xs font-semibold uppercase text-slate-500 tracking-wider">${title}</span>
         <span class="text-2xl font-bold ${colorClass}">${value}</span>
         <span class="text-xs text-slate-400">${sub}</span>
      </div>
    `;

    function setKpis(localStatus, trainMetrics){
      const g = trainMetrics.global_model?.latest || {};
      const kpisDiv = document.getElementById('kpis');
      const ver = localStatus.model_version;
      const rounds = localStatus.rounds;
      
      kpisDiv.innerHTML = 
        kpiCard('Global Round', '#' + rounds, 'Current Version: ' + ver, 'text-violet-400') +
        kpiCard('Total Samples', localStatus.samples_total.toLocaleString(), 'Combined fog set', 'text-white') +
        kpiCard('Global MAE', fmt(g.mae), 'Latest Validation', 'text-emerald-400') +
        kpiCard('Global RMSE', fmt(g.rmse), 'Latest Validation', 'text-emerald-400');
        
      // Sidebar status
      document.getElementById('statusBadges').innerHTML = `
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-900/50 text-blue-200 border border-blue-800">
           Personal Models: ${localStatus.personal_rounds}
        </span>
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-900/50 text-indigo-200 border border-indigo-800">
           Devices: ${localStatus.device_models?.length || 0}
        </span>
      `;
    }

    function renderTables(metrics){
      const pBody = document.getElementById('personalRows');
      pBody.innerHTML = '';
      const pList = metrics.personal_models?.latest_by_device || [];
      if(pList.length === 0) pBody.innerHTML = '<tr><td colspan="5" class="px-5 py-4 text-center text-slate-500 italic">No personal models yet</td></tr>';
      
      pList.forEach(r => {
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-800/30 transition-colors";
        tr.innerHTML = `
          <td class="px-5 py-3 font-medium text-white">${r.device_id}</td>
          <td class="px-5 py-3 text-slate-400">v${r.personal_version}</td>
          <td class="px-5 py-3 text-slate-400">${r.samples}</td>
          <td class="px-5 py-3 text-right font-mono text-emerald-400">${fmt(r.mae)}</td>
          <td class="px-5 py-3 text-right font-mono text-emerald-400">${fmt(r.rmse)}</td>`;
        pBody.appendChild(tr);
      });

      const gBody = document.getElementById('globalRows');
      gBody.innerHTML = '';
      const gList = metrics.global_model?.history || [];
      if(gList.length === 0) gBody.innerHTML = '<tr><td colspan="5" class="px-5 py-4 text-center text-slate-500 italic">No global history yet</td></tr>';
      
      gList.forEach(r => {
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-800/30 transition-colors";
        tr.innerHTML = `
          <td class="px-5 py-3 font-medium text-violet-300">#${r.round_version}</td>
          <td class="px-5 py-3 text-slate-400">${r.clients}</td>
          <td class="px-5 py-3 text-slate-400">${r.samples}</td>
          <td class="px-5 py-3 text-right font-mono text-emerald-400">${fmt(r.mae)}</td>
          <td class="px-5 py-3 text-right font-mono text-emerald-400">${fmt(r.rmse)}</td>`;
        gBody.appendChild(tr);
      });
    }

    function fillDeviceSelect(localStatus){
      const sel = document.getElementById('deviceSelect');
      const prev = sel.value;
      sel.innerHTML = '';
      const devs = localStatus.device_models || [];
      if(devs.length === 0) {
        const o = document.createElement('option');
        o.textContent = "No devices";
        sel.appendChild(o);
        return;
      }
      devs.forEach(d => {
        const o = document.createElement('option');
        o.value = d.device_id;
        o.textContent = `${d.device_id}`;
        o.className = "bg-slate-900";
        sel.appendChild(o);
      });
      if(prev && [...sel.options].some(o => o.value === prev)){ sel.value = prev; }
    }

    async function refreshAll(){
      const btn = document.getElementById('refreshBtn');
      const icon = btn.querySelector('svg');
      icon.classList.add('animate-spin');
      try {
        const [localStatus, trainMetrics] = await Promise.all([
          getJson('/federated/local-status'),
          getJson('/federated/training-metrics?limit=8')
        ]);
        setKpis(localStatus, trainMetrics);
        renderTables(trainMetrics);
        fillDeviceSelect(localStatus);
      } catch(e) {
        console.error(e);
      } finally {
        setTimeout(() => icon.classList.remove('animate-spin'), 500);
      }
    }

    async function loadEvaluation(){
      const device = document.getElementById('deviceSelect').value;
      const horizon = Number(document.getElementById('horizonSelect').value);
      if(!device || device === "No devices" || device === "Carregando...") return;
      
      const kDiv = document.getElementById('evalKpisInput');
      kDiv.innerHTML = '<div class="text-sm text-slate-500 animate-pulse">Computing metrics...</div>';
      
      try {
        const ev = await getJson(`/evaluation/series?device_id=${encodeURIComponent(device)}&horizon=${horizon}`);
        
        // Eval Stats
        if(!ev.ok){
          kDiv.innerHTML = '<div class="p-3 bg-red-900/20 border border-red-800 text-red-200 rounded text-sm">Insufficient data for this device.</div>';
          if(evalChart) { evalChart.destroy(); evalChart=null; }
          return;
        }
        
        const metricRow = (lbl, val, col) => `
          <div class="flex justify-between items-center p-2 rounded bg-slate-800/40 border border-slate-800">
            <span class="text-xs text-slate-400 uppercase">${lbl}</span>
            <span class="font-mono font-bold ${col}">${fmt(val)}</span>
          </div>
        `;
        
        kDiv.innerHTML = `
           <div class="grid grid-cols-2 gap-2 mb-2">
             ${metricRow('MAE', ev.metrics.mae, 'text-emerald-400')}
             ${metricRow('RMSE', ev.metrics.rmse, 'text-emerald-400')}
             ${metricRow('sMAPE', ev.metrics.smape, 'text-blue-400')}
             ${metricRow('Base MAE', (ev.baseline_naive||{}).mae, 'text-slate-500')}
           </div>
           <div class="text-xs text-slate-500 mt-2 flex gap-2">
              <span class="px-2 py-1 bg-slate-800 rounded border border-slate-700">Predictor: <span class="text-violet-300">${ev.selected_predictor}</span></span>
              <span class="px-2 py-1 bg-slate-800 rounded border border-slate-700">Source: <span class="text-violet-300">${ev.model_source}</span></span>
           </div>
        `;

        const actual = ev.series.actual || [];
        const forecast = ev.series.forecast || [];
        const labels = actual.map((_, i) => i + 1);

        if(evalChart){ evalChart.destroy(); }
        
        const ctx = document.getElementById('evalChart').getContext('2d');
        const gradReal = ctx.createLinearGradient(0,0,0,400);
        gradReal.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
        gradReal.addColorStop(1, 'rgba(16, 185, 129, 0)');
        
        const gradCast = ctx.createLinearGradient(0,0,0,400);
        gradCast.addColorStop(0, 'rgba(139, 92, 246, 0.4)');
        gradCast.addColorStop(1, 'rgba(139, 92, 246, 0)');

        evalChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels,
            datasets: [
              { 
                label: 'Real Value', 
                data: actual, 
                borderColor: '#10b981', 
                backgroundColor: gradReal,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1,
                fill: true
              },
              { 
                label: 'Prediction', 
                data: forecast, 
                borderColor: '#8b5cf6', 
                backgroundColor: gradCast,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
               legend: { position: 'top', align: 'end', labels: { boxWidth: 10, usePointStyle: true } },
               tooltip: { 
                 backgroundColor: 'rgba(15, 23, 42, 0.9)', 
                 titleColor: '#cbd5e1', 
                 bodyColor: '#e2e8f0', 
                 borderColor: '#334155', 
                 borderWidth: 1,
                 padding: 10
               }
            },
            scales: {
              x: { grid: { display: false } },
              y: { grid: { color: '#334155', drawBorder: false }, position: 'right' }
            }
          }
        });

      } catch(e){
        console.error(e);
        kDiv.innerHTML = `<span class="text-red-400 text-sm">Error: ${String(e)}</span>`;
      }
    }

    async function runSimulation(){
      const btn = document.getElementById('simulateBtn');
      const origText = btn.innerHTML;
      btn.innerHTML = `<svg class="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Simulation Running...`;
      btn.disabled = true;
      btn.classList.add('opacity-75', 'cursor-not-allowed');
      
      try {
        const body = {
          samples_per_profile: 250,
          sampling_hz: 30,
          seed: 20260225,
          run_global_training: true
        };
        await getJson('/federated/simulate-profiles', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        await refreshAll();
      } catch(e) {
        alert("Simulação falhou: " + e);
      } finally {
        btn.innerHTML = origText;
        btn.disabled = false;
        btn.classList.remove('opacity-75', 'cursor-not-allowed');
      }
    }

    document.getElementById('refreshBtn').addEventListener('click', refreshAll);
    document.getElementById('loadEvalBtn').addEventListener('click', loadEvaluation);
    document.getElementById('simulateBtn').addEventListener('click', runSimulation);

    refreshAll().catch(err => console.error(err));
  </script>
</body>
</html>
  """


@app.get("/", response_class=HTMLResponse)
def web_root() -> str:
  return """
<!doctype html>
<html><head><meta http-equiv=\"refresh\" content=\"0; url=/control/dashboard\" /></head>
<body style=\"background:#020617;color:#e2e8f0;font-family:Arial,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh\">
Redirecionando para Olivia Control Center...
</body></html>
  """


_init_db()
