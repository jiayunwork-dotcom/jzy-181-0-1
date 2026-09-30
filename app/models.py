"""Pydantic 请求/响应模型。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class UnitIn(BaseModel):
    name: str
    Pmin: float
    pmax: float
    a: float = 0.0
    b: float
    c: float
    ramp_up: float | list[float]
    ramp_down: float | list[float]
    initial_output: float


class LossCoeffs(BaseModel):
    B0: list[list[float]]
    B1: list[float] = Field(default_factory=lambda: [0.0])
    B2: float = 0.0


class FleetCreate(BaseModel):
    name: str
    units: list[UnitIn]
    loss: Optional[LossCoeffs] = None


class FleetUpdate(BaseModel):
    units: list[UnitIn]
    loss: Optional[LossCoeffs] = None


class ScheduleRequest(BaseModel):
    fleet_id: str
    loads: list[float]


class LocalRecomputeRequest(BaseModel):
    loads: list[float]
    changed_periods: Optional[list[int]] = None


class Created(BaseModel):
    id: str


class JobOut(BaseModel):
    job_id: str
    kind: str
    status: str
    progress: int
    result: Optional[dict] = None
    error: Optional[dict] = None
