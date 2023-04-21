from typing import Tuple
import autokoopman.core.trajectory as atraj
from sysidexpr.benchmark import Metric
import numpy as np


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
        y_diffs = t.states[:-1] ** 2.0
        idxs = np.logical_not(np.isnan(y_diffs.flatten()))
        traj_total = np.sum(dts[idxs] * y_diffs[idxs])
        totals.append(traj_total)

    return Metric(name="penalty_loss", lower_better=True), np.sqrt(np.sum(totals))
