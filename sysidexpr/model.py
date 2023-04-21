"""Pydantic schema for the sysidexpr model experiments"""
import pydantic
import pathlib
import os
from typing import List


class BenchmarkConfiguration(pydantic.BaseModel):
    """model of a benchmark of trajectories"""

    name: str
    data_csv: pathlib.Path
    prediction_dir: pathlib.Path
    states: List[str]
    groups: List[str]
    time: str
    traj: str

    @pydantic.validator("data_csv", pre=True)
    def data_csv_exists(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"path {v} is not a file")
        return v


class PredictionConfiguration(pydantic.BaseModel):
    """model to compare prediction trajectories against a benchmark"""

    model_name: str
    benchmark: BenchmarkConfiguration
    pred_csv: pathlib.Path

    @pydantic.validator("pred_csv", pre=True)
    def data_csv_exists(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"path {v} is not a file")
        return v


class Metric(pydantic.BaseModel):
    """information about a metric for scoring"""

    name: str
    lower_better: bool


class PredictionResult(pydantic.BaseModel):
    """prediction result model"""

    model_name: str
    benchmark_name: str
    metric: Metric
    value: float
