import os
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

FOG_URL = os.getenv("FOG_URL", "http://localhost:8002")

app = FastAPI(title="Edge HRV Service", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=False,
  allow_methods=["*"],
  allow_headers=["*"],
)


class HrvMetrics(BaseModel):
  rmssd: float = Field(ge=0)
  sdnn: float = Field(ge=0)
  pnn50: float = Field(ge=0)
  mean_hr: float = Field(ge=0)


class AnalyzePayload(BaseModel):
  user_id: str
  device_id: str
  sampling_hz: float = Field(gt=0)
  signal: list[float]
  metrics: HrvMetrics


@app.get("/health")
def health() -> dict[str, str]:
  return {"status": "ok", "layer": "edge"}


def local_stress_probability(metrics: HrvMetrics) -> float:
  hrv_score = (metrics.rmssd + metrics.sdnn + metrics.pnn50) / 3
  normalized_hrv = max(0.0, min(1.0, hrv_score / 80.0))
  normalized_hr = max(0.0, min(1.0, metrics.mean_hr / 120.0))
  probability = 0.65 * (1 - normalized_hrv) + 0.35 * normalized_hr
  return max(0.0, min(1.0, probability))


@app.post("/analyze")
async def analyze(payload: AnalyzePayload) -> dict[str, Any]:
  local_prob = local_stress_probability(payload.metrics)

  fog_prob = None
  try:
    async with httpx.AsyncClient(timeout=4.0) as client:
      response = await client.post(
        f"{FOG_URL}/risk/score",
        json={
          "user_id": payload.user_id,
          "device_id": payload.device_id,
          "metrics": payload.metrics.model_dump(),
          "local_probability": local_prob,
          "sampling_hz": payload.sampling_hz,
          "signal": payload.signal,
        },
      )
      response.raise_for_status()
      fog_prob = response.json().get("risk_probability")
  except Exception:
    fog_prob = None

  final_prob = float(fog_prob) if fog_prob is not None else local_prob
  high_stress = final_prob >= 0.70

  return {
    "user_id": payload.user_id,
    "device_id": payload.device_id,
    "stress_probability": final_prob,
    "high_stress": high_stress,
    "source": "fog" if fog_prob is not None else "edge-local",
  }
