from typing import Tuple

import autokoopman.core.trajectory as atraj
import numpy as np

from sysidexpr.benchmark import Metric
from typing import Optional


def _integration_loss(
    data: atraj.TrajectoriesData, pred: atraj.TrajectoriesData, n: Optional[int] = None
) -> Tuple[Metric, float]:
    """Compute the integration loss over a sample horizon"""
    # compute a trajectory where x_i = \|y_i - \hat{y}_i\|_2
    diff = (pred - data).norm()

    totals = []
    for t in diff:
        if len(t.times) < 2:
            continue

        # dealing with NaNs in eval is dangerous as predictor with
        # a lot of NaNs will score well
        dts = np.diff(t.times)
        y_diffs = t.states[1:] ** 2.0
        idxs = np.logical_not(np.isnan(y_diffs.flatten()))

        # slice idxs over a fixed time horizon
        if n is not None:
            idxs = idxs[:n]

        traj_total = np.sum(dts[idxs] * y_diffs[idxs])
        totals.append(traj_total)

        # warn if there are NaNs in the trajectory
        if np.any(np.isnan(totals)):
            # print a warning
            print("WARNING: NaNs in penalty loss")

    return Metric(name="penalty_loss", lower_better=True), np.sqrt(np.sum(totals))


def integration_loss_5(
    data: atraj.TrajectoriesData, pred: atraj.TrajectoriesData
) -> Tuple[Metric, float]:
    """Compute the integration loss over a 10 samples horizon"""
    return _integration_loss(data, pred, n=5)


def integration_loss_10(
    data: atraj.TrajectoriesData, pred: atraj.TrajectoriesData
) -> Tuple[Metric, float]:
    """Compute the integration loss over a 10 samples horizon"""
    return _integration_loss(data, pred, n=10)


def integration_loss(
    data: atraj.TrajectoriesData, pred: atraj.TrajectoriesData
) -> Tuple[Metric, float]:
    """Compute the integration loss over the entire horizon"""
    return _integration_loss(data, pred)


def penalty_loss(
    data: atraj.TrajectoriesData, pred: atraj.TrajectoriesData
) -> Tuple[Metric, float]:
    """metric from RKHS JCPX submission"""
    # compute a trajectory where x_i = \|y_i - \hat{y}_i\|_2
    diff = (pred - data).norm()

    totals = []
    for t in diff:
        if len(t.times) < 2:
            continue

        # dealing with NaNs in eval is dangerous as predictor with
        # a lot of NaNs will score well
        dts = np.diff(t.times)
        y_diffs = t.states[1:] ** 2.0
        idxs = np.logical_not(np.isnan(y_diffs.flatten()))
        traj_total = np.sum(dts[idxs] * y_diffs[idxs])
        totals.append(traj_total)

        # warn if there are NaNs in the trajectory
        if np.any(np.isnan(totals)):
            # print a warning
            print("WARNING: NaNs in penalty loss")

    return Metric(name="penalty_loss", lower_better=True), np.sqrt(np.sum(totals))
