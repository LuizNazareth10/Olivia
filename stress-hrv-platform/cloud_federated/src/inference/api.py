from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import threading
import time
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
  import torch
except Exception:  # pragma: no cover
  torch = None

app = FastAPI(title="Cloud Federated HRV API", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "federated_store.db"
MODEL_PATH = BASE_DIR / "models" / "global_model.pt"

MIN_CLIENTS_PER_ROUND = 2
MIN_SAMPLES_PER_CLIENT = 5
MAX_SAMPLES_PER_CLIENT = 64
LOCAL_EPOCHS = 1

_LOCK = threading.Lock()

_MODEL = None
if torch is not None:
  try:
    from src.models.bilstm_attention import BiLSTMAttention

    _MODEL = BiLSTMAttention()
    _MODEL.eval()

    if MODEL_PATH.exists():
      state = torch.load(MODEL_PATH, map_location="cpu")
      _MODEL.load_state_dict(state)
  except Exception:
    _MODEL = None


class PredictPayload(BaseModel):
  user_id: str
  metrics: dict[str, float]
  sampling_hz: float = Field(gt=0)


class IngestPayload(BaseModel):
  device_id: str
  user_id: str
  metrics: dict[str, float]
  sampling_hz: float = Field(gt=0)
  local_probability: float = Field(ge=0, le=1)
  final_probability: float = Field(ge=0, le=1)
  high_stress: bool
  source_model: str = "edge-fog-cloud"


class ModelPushPayload(BaseModel):
  fog_id: str
  round_version: int
  clients: int
  samples: int
  participant_aliases: list[str] = Field(default_factory=list)
  cohort_key: str | None = None
  metrics: dict[str, float | None]
  state_b64: str


_EVAL_CACHE: dict[str, Any] = {"ts": 0.0, "summary": None}


@app.get("/health")
def health() -> dict[str, str]:
  return {"status": "ok", "layer": "cloud"}


def _get_conn() -> sqlite3.Connection:
  DB_DIR.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn


def _init_db() -> None:
  with _get_conn() as conn:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS captures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        device_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        sampling_hz REAL NOT NULL,
        rmssd REAL NOT NULL,
        sdnn REAL NOT NULL,
        pnn50 REAL NOT NULL,
        mean_hr REAL NOT NULL,
        signal_json TEXT NOT NULL,
        local_probability REAL NOT NULL,
        final_probability REAL NOT NULL,
        high_stress INTEGER NOT NULL,
        source_model TEXT NOT NULL,
        trained_round INTEGER
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS device_model_state (
        device_id TEXT PRIMARY KEY,
        samples_seen INTEGER NOT NULL DEFAULT 0,
        last_model_version INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS federated_rounds (
        round_version INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        clients INTEGER NOT NULL,
        samples INTEGER NOT NULL
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS coordinator_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        fog_id TEXT NOT NULL,
        round_version INTEGER NOT NULL,
        clients INTEGER NOT NULL,
        samples INTEGER NOT NULL,
        mae REAL,
        rmse REAL,
        smape REAL,
        mase REAL,
        participant_aliases_json TEXT,
        cohort_key TEXT
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
    conn.execute(
      "INSERT OR IGNORE INTO metadata(key, value) VALUES('model_version', '0')"
    )
    existing_cols = {
      row["name"]
      for row in conn.execute("PRAGMA table_info(coordinator_rounds)").fetchall()
    }
    if "participant_aliases_json" not in existing_cols:
      conn.execute("ALTER TABLE coordinator_rounds ADD COLUMN participant_aliases_json TEXT")
    if "cohort_key" not in existing_cols:
      conn.execute("ALTER TABLE coordinator_rounds ADD COLUMN cohort_key TEXT")
    conn.commit()


def _get_model_version(conn: sqlite3.Connection) -> int:
  row = conn.execute("SELECT value FROM metadata WHERE key='model_version'").fetchone()
  if row is None:
    conn.execute("INSERT INTO metadata(key, value) VALUES('model_version', '0')")
    conn.commit()
    return 0
  return int(row["value"])


def _set_model_version(conn: sqlite3.Connection, version: int) -> None:
  conn.execute(
    "UPDATE metadata SET value=? WHERE key='model_version'",
    (str(version),),
  )


def _build_training_tensors(rows: list[sqlite3.Row]) -> tuple[Any, Any]:
  assert torch is not None
  features = []
  labels = []
  for row in rows:
    rmssd = float(row["rmssd"])
    sdnn = float(row["sdnn"])
    pnn50 = float(row["pnn50"])
    mean_hr = float(row["mean_hr"])
    features.append([[rmssd, sdnn, pnn50, mean_hr]] * 30)
    labels.append(float(row["high_stress"]))

  x = torch.tensor(features, dtype=torch.float32)
  y = torch.tensor(labels, dtype=torch.float32)
  return x, y


def _clone_state_dict(model) -> dict[str, Any]:
  return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _fedavg(states: list[dict[str, Any]]) -> dict[str, Any]:
  avg = {}
  for key in states[0].keys():
    avg[key] = torch.stack([state[key] for state in states], dim=0).mean(dim=0)
  return avg


def _local_train(initial_state: dict[str, Any], rows: list[sqlite3.Row]) -> dict[str, Any]:
  assert torch is not None
  if _MODEL is None:
    raise RuntimeError("Global model unavailable")

  from src.models.bilstm_attention import BiLSTMAttention

  model = BiLSTMAttention()
  model.load_state_dict(initial_state)
  model.train()

  x, y = _build_training_tensors(rows)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.BCELoss()

  for _ in range(LOCAL_EPOCHS):
    pred = model(x)
    loss = criterion(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

  return _clone_state_dict(model)


def _try_run_federated_round(conn: sqlite3.Connection) -> dict[str, Any]:
  if _MODEL is None or torch is None:
    return {"trained": False, "reason": "model-unavailable"}

  device_rows = conn.execute(
    """
    SELECT device_id, COUNT(*) AS cnt
    FROM captures
    WHERE trained_round IS NULL
    GROUP BY device_id
    HAVING cnt >= ?
    """,
    (MIN_SAMPLES_PER_CLIENT,),
  ).fetchall()

  if len(device_rows) < MIN_CLIENTS_PER_ROUND:
    return {
      "trained": False,
      "reason": "insufficient-clients",
      "eligible_clients": len(device_rows),
    }

  client_ids = [row["device_id"] for row in device_rows]
  current_version = _get_model_version(conn)
  next_version = current_version + 1

  global_state = _clone_state_dict(_MODEL)
  local_states = []
  used_capture_ids: list[int] = []

  for device_id in client_ids:
    rows = conn.execute(
      """
      SELECT *
      FROM captures
      WHERE device_id = ? AND trained_round IS NULL
      ORDER BY id ASC
      LIMIT ?
      """,
      (device_id, MAX_SAMPLES_PER_CLIENT),
    ).fetchall()
    if len(rows) < MIN_SAMPLES_PER_CLIENT:
      continue

    local_states.append(_local_train(global_state, rows))
    used_capture_ids.extend([int(row["id"]) for row in rows])

  if len(local_states) < MIN_CLIENTS_PER_ROUND:
    return {
      "trained": False,
      "reason": "insufficient-clients-after-filter",
      "eligible_clients": len(local_states),
    }

  new_state = _fedavg(local_states)
  _MODEL.load_state_dict(new_state)
  _MODEL.eval()

  MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
  torch.save(_MODEL.state_dict(), MODEL_PATH)

  placeholders = ",".join(["?"] * len(used_capture_ids))
  conn.execute(
    f"UPDATE captures SET trained_round = ? WHERE id IN ({placeholders})",
    (next_version, *used_capture_ids),
  )

  for device_id in client_ids:
    sample_count = conn.execute(
      "SELECT COUNT(*) AS cnt FROM captures WHERE device_id = ?",
      (device_id,),
    ).fetchone()["cnt"]

    conn.execute(
      """
      INSERT INTO device_model_state(device_id, samples_seen, last_model_version, updated_at)
      VALUES(?, ?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(device_id) DO UPDATE SET
        samples_seen=excluded.samples_seen,
        last_model_version=excluded.last_model_version,
        updated_at=CURRENT_TIMESTAMP
      """,
      (device_id, int(sample_count), next_version),
    )

  conn.execute(
    "INSERT INTO federated_rounds(round_version, clients, samples) VALUES(?, ?, ?)",
    (next_version, len(local_states), len(used_capture_ids)),
  )
  _set_model_version(conn, next_version)
  conn.commit()

  return {
    "trained": True,
    "round_version": next_version,
    "clients": len(local_states),
    "samples": len(used_capture_ids),
  }


def _heuristic_probability(metrics: dict[str, float]) -> float:
  rmssd = metrics.get("rmssd", 0.0)
  sdnn = metrics.get("sdnn", 0.0)
  pnn50 = metrics.get("pnn50", 0.0)
  mean_hr = metrics.get("mean_hr", 0.0)

  hrv_score = (rmssd + sdnn + pnn50) / 3
  normalized_hrv = max(0.0, min(1.0, hrv_score / 80.0))
  normalized_hr = max(0.0, min(1.0, mean_hr / 120.0))
  probability = 0.65 * (1 - normalized_hrv) + 0.35 * normalized_hr
  return max(0.0, min(1.0, probability))


@app.post("/predict")
def predict(payload: PredictPayload):
  if _MODEL is not None and torch is not None:
    rmssd = payload.metrics.get("rmssd", 0.0)
    sdnn = payload.metrics.get("sdnn", 0.0)
    pnn50 = payload.metrics.get("pnn50", 0.0)
    mean_hr = payload.metrics.get("mean_hr", 0.0)

    features = torch.tensor([[[rmssd, sdnn, pnn50, mean_hr]] * 30], dtype=torch.float32)

    with torch.no_grad():
      prob = float(_MODEL(features).item())
    model_name = "global-bilstm-attention"
  else:
    prob = _heuristic_probability(payload.metrics)
    model_name = "heuristic-fallback"

  prob = max(0.0, min(1.0, prob))

  with _get_conn() as conn:
    model_version = _get_model_version(conn)

  return {
    "user_id": payload.user_id,
    "stress_probability": prob,
    "high_stress": prob >= 0.70,
    "model": model_name,
    "model_version": model_version,
  }


@app.post("/federated/ingest")
def federated_ingest(payload: IngestPayload):
  rmssd = float(payload.metrics.get("rmssd", 0.0))
  sdnn = float(payload.metrics.get("sdnn", 0.0))
  pnn50 = float(payload.metrics.get("pnn50", 0.0))
  mean_hr = float(payload.metrics.get("mean_hr", 0.0))

  with _LOCK:
    with _get_conn() as conn:
      conn.execute(
        """
        INSERT INTO captures(
          device_id, user_id, sampling_hz, rmssd, sdnn, pnn50, mean_hr,
          signal_json, local_probability, final_probability, high_stress, source_model
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          payload.device_id,
          payload.user_id,
          payload.sampling_hz,
          rmssd,
          sdnn,
          pnn50,
          mean_hr,
          "[]",
          payload.local_probability,
          payload.final_probability,
          1 if payload.high_stress else 0,
          payload.source_model,
        ),
      )

      sample_count = conn.execute(
        "SELECT COUNT(*) AS cnt FROM captures WHERE device_id = ?",
        (payload.device_id,),
      ).fetchone()["cnt"]

      current_version = _get_model_version(conn)
      conn.execute(
        """
        INSERT INTO device_model_state(device_id, samples_seen, last_model_version, updated_at)
        VALUES(?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(device_id) DO UPDATE SET
          samples_seen=excluded.samples_seen,
          updated_at=CURRENT_TIMESTAMP
        """,
        (payload.device_id, int(sample_count), current_version),
      )
      conn.commit()

      train_result = _try_run_federated_round(conn)

      model_version = _get_model_version(conn)
      pending = conn.execute(
        "SELECT COUNT(*) AS cnt FROM captures WHERE trained_round IS NULL"
      ).fetchone()["cnt"]

  return {
    "ok": True,
    "device_id": payload.device_id,
    "samples_for_device": int(sample_count),
    "pending_samples": int(pending),
    "model_version": int(model_version),
    "train_result": train_result,
  }


@app.get("/federated/status")
def federated_status():
  with _get_conn() as conn:
    model_version = _get_model_version(conn)
    rounds = conn.execute("SELECT COUNT(*) AS cnt FROM coordinator_rounds").fetchone()["cnt"]
    samples_total = conn.execute("SELECT COUNT(*) AS cnt FROM captures").fetchone()["cnt"]
    pending = conn.execute("SELECT COUNT(*) AS cnt FROM captures WHERE trained_round IS NULL").fetchone()["cnt"]
    last_round = conn.execute(
      """
      SELECT fog_id, round_version, clients, samples, mae, rmse, smape, mase,
             participant_aliases_json, cohort_key, created_at
      FROM coordinator_rounds
      ORDER BY id DESC LIMIT 1
      """
    ).fetchone()

  parsed_last_round = None
  if last_round is not None:
    parsed_last_round = dict(last_round)
    aliases_raw = parsed_last_round.pop("participant_aliases_json", None)
    try:
      aliases = json.loads(aliases_raw) if aliases_raw else []
    except Exception:
      aliases = []
    parsed_last_round["participant_aliases"] = aliases
    parsed_last_round["participants"] = len(aliases)
    if not parsed_last_round.get("cohort_key"):
      parsed_last_round["cohort_key"] = (
        "COHORT-" + hashlib.sha256("|".join(sorted(aliases)).encode("utf-8")).hexdigest()[:12].upper()
      ) if aliases else None

  return {
    "model_version": int(model_version),
    "rounds": int(rounds),
    "legacy_samples_total": int(samples_total),
    "legacy_pending_samples": int(pending),
    "last_round": parsed_last_round,
    "mode": "federated-strict-no-user-data",
    "device_ids_visibility": "hidden-in-cloud",
    "identifier_strategy": "anonymized-participant-aliases",
    "device_ids_source": "http://127.0.0.1:8002/federated/local-status",
    "db_path": str(DB_PATH),
  }


@app.post("/federated/model/push")
def federated_model_push(payload: ModelPushPayload):
  if _MODEL is None or torch is None:
    return {"ok": False, "reason": "model-unavailable"}

  state_bytes = base64.b64decode(payload.state_b64.encode("utf-8"))
  state = torch.load(io.BytesIO(state_bytes), map_location="cpu")
  _MODEL.load_state_dict(state)
  _MODEL.eval()
  MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
  torch.save(_MODEL.state_dict(), MODEL_PATH)

  with _LOCK:
    with _get_conn() as conn:
      current_version = _get_model_version(conn)
      next_version = max(current_version + 1, int(payload.round_version))
      _set_model_version(conn, next_version)
      conn.execute(
        """
        INSERT INTO coordinator_rounds(
          fog_id, round_version, clients, samples, mae, rmse, smape, mase,
          participant_aliases_json, cohort_key
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          payload.fog_id,
          int(payload.round_version),
          int(payload.clients),
          int(payload.samples),
          payload.metrics.get("mae"),
          payload.metrics.get("rmse"),
          payload.metrics.get("smape"),
          payload.metrics.get("mase"),
          json.dumps(payload.participant_aliases),
          payload.cohort_key,
        ),
      )
      conn.commit()

  return {
    "ok": True,
    "fog_id": payload.fog_id,
    "accepted_round_version": int(payload.round_version),
    "model_version": int(next_version),
  }


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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, in_sample: np.ndarray) -> dict[str, float]:
  mae = float(np.mean(np.abs(y_true - y_pred)))
  rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
  return {
    "mae": mae,
    "rmse": rmse,
    "smape": _smape(y_true, y_pred),
    "mase": _mase(y_true, y_pred, in_sample),
  }


def _safe_number(value: float) -> float | None:
  if value is None:
    return None
  num = float(value)
  if math.isnan(num) or math.isinf(num):
    return None
  return num


def _safe_metric_dict(metrics: dict[str, float]) -> dict[str, float | None]:
  return {k: _safe_number(float(v)) for k, v in metrics.items()}


def _device_rows(conn: sqlite3.Connection, device_id: str) -> list[sqlite3.Row]:
  return conn.execute(
    """
    SELECT id, created_at, rmssd, sdnn, pnn50, mean_hr, final_probability
    FROM captures
    WHERE device_id = ?
    ORDER BY id ASC
    """,
    (device_id,),
  ).fetchall()


def _build_sequences(rows: list[sqlite3.Row], horizon: int, seq_len: int = 30):
  if len(rows) < seq_len + horizon + 5:
    return None

  feat = np.array(
    [[float(r["rmssd"]), float(r["sdnn"]), float(r["pnn50"]), float(r["mean_hr"])] for r in rows],
    dtype=np.float32,
  )
  target = np.array([float(r["final_probability"]) for r in rows], dtype=np.float32)
  stamps = [str(r["created_at"]) for r in rows]

  x_list = []
  y_list = []
  t_list = []
  for i in range(seq_len, len(rows) - horizon + 1):
    x_list.append(feat[i - seq_len:i])
    y_list.append(target[i + horizon - 1])
    t_list.append(stamps[i + horizon - 1])

  x = np.array(x_list, dtype=np.float32)
  y = np.array(y_list, dtype=np.float32)
  if len(x) < 20:
    return None

  split = max(int(len(x) * 0.7), 10)
  split = min(split, len(x) - 5)
  x_train = x[:split]
  y_train = y[:split]
  x_test = x[split:]
  y_test = y[split:]
  t_test = t_list[split:]

  mean = x_train.mean(axis=(0, 1), keepdims=True)
  std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
  x_train = (x_train - mean) / std
  x_test = (x_test - mean) / std

  return x_train, y_train, x_test, y_test, t_test


def _bilstm_eval_for_horizon(rows: list[sqlite3.Row], horizon: int) -> dict[str, Any] | None:
  if torch is None:
    return None
  if _MODEL is None:
    return None

  data = _build_sequences(rows, horizon=horizon)
  if data is None:
    return None

  x_train, y_train, x_test, y_test, t_test = data
  if len(x_test) == 0:
    return None

  from src.models.bilstm_attention import BiLSTMAttention

  model = BiLSTMAttention(input_dim=4, hidden_dim=32, layers=1, dropout=0.1)
  model.train()

  x_train_t = torch.tensor(x_train, dtype=torch.float32)
  y_train_t = torch.tensor(y_train, dtype=torch.float32)
  x_test_t = torch.tensor(x_test, dtype=torch.float32)

  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.MSELoss()

  for _ in range(18):
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

  resid_std = float(np.std(y_train - pred_train))
  interval_low = np.clip(pred_test - 1.96 * resid_std, 0.0, 1.0)
  interval_high = np.clip(pred_test + 1.96 * resid_std, 0.0, 1.0)
  coverage = float(np.mean((y_test >= interval_low) & (y_test <= interval_high)))
  avg_width = float(np.mean(interval_high - interval_low))

  metric = _metrics(y_test, pred_test, y_train)
  return {
    "horizon": horizon,
    "metrics": metric,
    "series": {
      "timestamps": t_test,
      "actual": y_test.tolist(),
      "forecast": pred_test.tolist(),
      "lower": interval_low.tolist(),
      "upper": interval_high.tolist(),
    },
    "interval": {
      "coverage": coverage,
      "avg_width": avg_width,
    },
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
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(math.sqrt(float(np.mean((a - p) ** 2))))
    result.append({"window": i + 1, "mae": mae, "rmse": rmse})
  return result


def _evaluate_device(conn: sqlite3.Connection, device_id: str) -> dict[str, Any] | None:
  rows = _device_rows(conn, device_id)
  if len(rows) < 40:
    return None

  horizons = [1, 7, 30]
  horizon_eval: dict[str, Any] = {}
  horizon_errors = []
  first_series = None
  intervals = {}

  for h in horizons:
    out = _bilstm_eval_for_horizon(rows, horizon=h)
    if out is None:
      continue
    horizon_eval[f"t+{h}"] = out["metrics"]
    horizon_errors.append(
      {
        "horizon": f"t+{h}",
        "mae": _safe_number(out["metrics"]["mae"]),
        "rmse": _safe_number(out["metrics"]["rmse"]),
      }
    )
    intervals[f"t+{h}"] = out["interval"]
    if h == 1:
      first_series = out["series"]

  if not horizon_eval:
    return None

  overall = horizon_eval.get("t+1") or next(iter(horizon_eval.values()))
  backtesting = []
  if first_series is not None:
    backtesting = _backtesting_windows(first_series["actual"], first_series["forecast"])

  return {
    "device_id": device_id,
    "samples": len(rows),
    "overall": _safe_metric_dict(overall),
    "horizons": {k: _safe_metric_dict(v) for k, v in horizon_eval.items()},
    "errors_by_horizon": horizon_errors,
    "intervals": intervals,
    "series_t1": first_series,
    "backtesting": backtesting,
  }


def _compute_evaluation_summary(force: bool = False) -> dict[str, Any]:
  now = time.time()
  if not force and _EVAL_CACHE.get("summary") is not None and now - float(_EVAL_CACHE.get("ts", 0)) < 30:
    return _EVAL_CACHE["summary"]

  with _get_conn() as conn:
    devices = [
      row["device_id"]
      for row in conn.execute("SELECT DISTINCT device_id FROM captures ORDER BY device_id").fetchall()
    ]
    per_device = []
    for d in devices:
      ev = _evaluate_device(conn, d)
      if ev is not None:
        per_device.append(ev)

    if per_device:
      mae_vals = [d["overall"]["mae"] for d in per_device if not math.isnan(d["overall"]["mae"])]
      rmse_vals = [d["overall"]["rmse"] for d in per_device if not math.isnan(d["overall"]["rmse"])]
      smape_vals = [d["overall"]["smape"] for d in per_device if not math.isnan(d["overall"]["smape"])]
      mase_vals = [d["overall"]["mase"] for d in per_device if not math.isnan(d["overall"]["mase"])]
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
      "devices": [
        {
          "device_id": d["device_id"],
          "samples": d["samples"],
          "overall": d["overall"],
          "horizons": d["horizons"],
        }
        for d in per_device
      ],
    }

  _EVAL_CACHE["ts"] = now
  _EVAL_CACHE["summary"] = out
  return out


@app.get("/evaluation/summary")
def evaluation_summary(force: bool = Query(default=False)):
  return _compute_evaluation_summary(force=force)


@app.get("/evaluation/series")
def evaluation_series(device_id: str, horizon: int = Query(default=1, ge=1, le=30)):
  with _get_conn() as conn:
    rows = _device_rows(conn, device_id)
    out = _bilstm_eval_for_horizon(rows, horizon=horizon)
    if out is None:
      return {
        "ok": False,
        "message": "Dados insuficientes para avaliar este device/horizonte (mínimo recomendado: ~40 amostras por device).",
      }

    series = out["series"]
    return {
      "ok": True,
      "device_id": device_id,
      "horizon": horizon,
      "metrics": _safe_metric_dict(out["metrics"]),
      "error_by_horizon": {
        "horizon": f"t+{horizon}",
        "mae": _safe_number(out["metrics"]["mae"]),
        "rmse": _safe_number(out["metrics"]["rmse"]),
      },
      "series": series,
      "backtesting": _backtesting_windows(series["actual"], series["forecast"]),
      "interval": out["interval"],
    }


@app.get("/evaluation/dashboard", response_class=HTMLResponse)
@app.get("/cloud/dashboard", response_class=HTMLResponse)
def evaluation_dashboard():
  return """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation | Cloud</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .glass {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .glass-header {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
    <nav class="glass-header fixed top-0 w-full z-50">
        <div class="px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-fuchsia-600 flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                    </svg>
                </div>
                <span class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-fuchsia-400">Cloud Evaluation</span>
            </div>
            <div class="flex items-center space-x-4">
                <button onclick="refreshData()" class="px-4 py-2 text-sm bg-violet-900/40 hover:bg-violet-900/60 text-violet-300 rounded-lg transition-colors border border-violet-700/50">
                    Refresh Metrics
                </button>
            </div>
        </div>
    </nav>

    <main class="pt-24 px-6 pb-12 max-w-7xl mx-auto space-y-6">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div class="glass rounded-xl p-6 relative overflow-hidden group">
                <div class="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <h3 class="text-slate-400 text-sm font-medium mb-2">Global MAE</h3>
                <div class="text-3xl font-bold text-blue-400" id="global-mae">--</div>
            </div>
            <div class="glass rounded-xl p-6 relative overflow-hidden group">
                <div class="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <h3 class="text-slate-400 text-sm font-medium mb-2">Global RMSE</h3>
                <div class="text-3xl font-bold text-emerald-400" id="global-rmse">--</div>
            </div>
            <div class="glass rounded-xl p-6 relative overflow-hidden group">
                <div class="absolute inset-0 bg-gradient-to-br from-amber-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <h3 class="text-slate-400 text-sm font-medium mb-2">Devices</h3>
                <div class="text-3xl font-bold text-amber-400" id="device-count">--</div>
            </div>
            <div class="glass rounded-xl p-6 relative overflow-hidden group">
                <div class="absolute inset-0 bg-gradient-to-br from-violet-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <h3 class="text-slate-400 text-sm font-medium mb-2">Generated</h3>
                <div class="text-lg font-mono text-violet-400 truncate" id="generated-at">--</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="glass rounded-xl p-6 lg:col-span-1 h-[600px] flex flex-col">
                <h3 class="text-lg font-bold text-slate-100 mb-4">Devices</h3>
                <div class="overflow-y-auto flex-1 pr-2 space-y-2 custom-scrollbar" id="device-list">
                    <div class="text-center text-slate-500 py-10">Loading...</div>
                </div>
            </div>

            <div class="glass rounded-xl p-6 lg:col-span-2 flex flex-col">
                <div class="flex justify-between items-center mb-6">
                    <h3 class="text-lg font-bold text-slate-100" id="selected-device-title">Select Device</h3>
                    <select id="horizon-select" class="bg-slate-800 border border-slate-700 text-sm rounded-lg px-3 py-1 ring-1 ring-slate-700 focus:ring-violet-500 outline-none text-slate-300">
                        <option value="1">Horizon t+1</option>
                        <option value="7">Horizon t+7</option>
                        <option value="30">Horizon t+30</option>
                    </select>
                </div>
                <div class="flex-1 relative min-h-[400px]">
                    <canvas id="evalChart"></canvas>
                </div>
                <div class="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-slate-700/50">
                     <div class="text-center">
                        <div class="text-xs text-slate-500">Device MAE</div>
                        <div class="font-bold text-blue-400" id="dev-mae">--</div>
                    </div>
                    <div class="text-center">
                        <div class="text-xs text-slate-500">Device RMSE</div>
                        <div class="font-bold text-emerald-400" id="dev-rmse">--</div>
                    </div>
                    <div class="text-center">
                        <div class="text-xs text-slate-500">Coverage</div>
                        <div class="font-bold text-amber-400" id="dev-cov">--</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let currentChart = null;
        let selectedDevice = null;
        
        // Colors for Chart.js in Dark Mode
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = '#334155';

        async function refreshData() {
            try {
                const res = await fetch('/evaluation/summary?force=true');
                const data = await res.json();
                renderSummary(data);
            } catch (e) {
                console.error(e);
            }
        }

        function renderSummary(data) {
            document.getElementById('global-mae').textContent = data.global_overall.mae?.toFixed(4) || '--';
            document.getElementById('global-rmse').textContent = data.global_overall.rmse?.toFixed(4) || '--';
            document.getElementById('device-count').textContent = data.devices_with_eval || 0;
            document.getElementById('generated-at').textContent = new Date(data.generated_at * 1000).toLocaleTimeString();

            const list = document.getElementById('device-list');
            list.innerHTML = '';
            
            if (data.devices.length === 0) list.innerHTML = '<div class="text-center text-slate-500">No devices</div>';

            data.devices.forEach(d => {
                const el = document.createElement('div');
                el.className = 'p-3 rounded-lg bg-slate-800/50 hover:bg-slate-700 cursor-pointer transition-colors border border-slate-700/50 hover:border-violet-500/30';
                el.onclick = () => loadDeviceDetails(d.device_id);
                el.innerHTML = `
                    <div class="flex justify-between items-center mb-1">
                        <span class="font-medium text-slate-300 text-sm truncate">${d.device_id.substring(0, 10)}...</span>
                        <span class="text-xs text-violet-400">${d.samples} pts</span>
                    </div>
                `;
                list.appendChild(el);
            });

            if (!selectedDevice && data.devices.length > 0) loadDeviceDetails(data.devices[0].device_id);
        }

        async function loadDeviceDetails(deviceId) {
            selectedDevice = deviceId;
            const horizon = document.getElementById('horizon-select').value;
            document.getElementById('selected-device-title').textContent = `Device: ${deviceId}`;
            
            const res = await fetch(`/evaluation/series?device_id=${deviceId}&horizon=${horizon}`);
            const data = await res.json();
            
            if (data.ok) {
                renderChart(data);
                 document.getElementById('dev-mae').textContent = data.metrics.mae?.toFixed(4) || '--';
                document.getElementById('dev-rmse').textContent = data.metrics.rmse?.toFixed(4) || '--';
                document.getElementById('dev-cov').textContent = (data.interval.coverage * 100).toFixed(1) + '%' || '--';
            }
        }

        function renderChart(data) {
            const ctx = document.getElementById('evalChart').getContext('2d');
            if (currentChart) currentChart.destroy();

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.series.timestamps.map(t => new Date(t).toLocaleTimeString()),
                    datasets: [
                        { label: 'Actual', data: data.series.actual, borderColor: '#3b82f6', backgroundColor: '#3b82f6', tension: 0.1 },
                        { label: 'Forecast', data: data.series.forecast, borderColor: '#a855f7', borderDash: [5,5], tension: 0.1 },
                        { label: 'Upper', data: data.series.upper, borderColor: 'transparent', backgroundColor: 'rgba(168, 85, 247, 0.1)', fill: '+1' },
                        { label: 'Lower', data: data.series.lower, borderColor: 'transparent', fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { display: false }
                    },
                    plugins: { legend: { labels: { color: '#94a3b8' } } }
                }
            });
        }

        document.getElementById('horizon-select').addEventListener('change', () => {
            if (selectedDevice) loadDeviceDetails(selectedDevice);
        });

        refreshData();
    </script>
</body>
</html>
  """


_init_db()
