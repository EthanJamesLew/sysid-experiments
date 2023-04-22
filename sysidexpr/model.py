"""Pydantic schema for the sysidexpr model experiments"""
import os
import pathlib
from typing import List

import pydantic


class BenchmarkConfiguration(pydantic.BaseModel):
    """model of a benchmark of trajectories"""

    name: str
    data_csv: pathlib.Path
    prediction_dir: pathlib.Path
    states: List[str]
    groups: List[str]
    time: str
    traj: str


class BenchmarkSchema(pydantic.BaseModel):
    """list of benchmarks"""
    benchmarks: List[BenchmarkConfiguration]


class PredictionConfiguration(pydantic.BaseModel):
    """model to compare prediction trajectories against a benchmark"""

    model_name: str
    benchmark: BenchmarkConfiguration
    pred_csv: pathlib.Path


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
