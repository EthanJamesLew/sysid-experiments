""" benchmarking utilities for sysidexpr """
import json
import os
import pathlib
from typing import Callable
from typing import Tuple

import autokoopman.core.trajectory as atraj
import numpy as np
import pandas as pd
from pydantic import parse_obj_as

from sysidexpr.model import BenchmarkConfiguration
from sysidexpr.model import BenchmarkSchema
from sysidexpr.model import Metric
from sysidexpr.model import PredictionConfiguration
from sysidexpr.model import PredictionResult


def load_benchmark_configs(schema_path: pathlib.Path) -> BenchmarkSchema:
    """load the benchmark schema from a json file"""
    assert os.path.isfile(schema_path)
    # open the json and load into the relevant config models
    with open(schema_path, "r") as fp:
        benchmarks_raw = json.load(fp)
    return parse_obj_as(BenchmarkSchema, benchmarks_raw)


class Benchmark:
    """benchmark loader"""

    def __init__(self, config: BenchmarkConfiguration):
        self.config = config

    def load_trajectories(self):
        # put the csv contents into a pandas frame
        data_df = pd.read_csv(self.config.data_csv)

        def extract_time_states(df):
            """pack the states and times into a autokoopman trajectory"""
            times = df[self.config.time].to_numpy().flatten()
            idxs = np.argsort(times)
            states = df[self.config.states].to_numpy()
            return atraj.Trajectory(
                times=times[idxs],
                states=states[idxs],
                inputs=None,
                state_names=self.config.states,
            )

        # group by subject id
        subjects_dfs = data_df.groupby(data_df[self.config.traj])

        # build them into a TrajectoriesData
        trajectories = {g: {} for g in self.config.groups}
        for groupname in self.config.groups:
            for sid, s_df in subjects_dfs:
                # candidate may not be in these groups
                candidate_df = s_df.loc[s_df[groupname] == True]
                if len(candidate_df) > 0:
                    trajectories[groupname][sid] = extract_time_states(candidate_df)

        return {g: atraj.TrajectoriesData(ts) for g, ts in trajectories.items()}

    def store_trajectories(self, pred_trajs, group_name="Test") -> pd.DataFrame:
        """store the predicted trajectories in a DataFrame"""
        # create the DataFrame columns
        columns = [self.config.traj, self.config.time, *self.config.states, group_name]

        # add rows per trajectory id
        traj_rows = []
        for tname, traj in pred_trajs._trajs.items():
            data = np.hstack(
                (
                    [
                        [tname],
                    ]
                    * len(traj.times),
                    np.atleast_2d(traj.times).T,
                    traj.states,
                    [
                        [
                            True,
                        ],
                    ]
                    * len(traj.times),
                )
            )
            traj_rows.append(data)

        # stack to create a DataFrame
        return pd.DataFrame(columns=columns, data=np.vstack(traj_rows))

    @staticmethod
    def score_benchmark(
        prediction: PredictionConfiguration,
        scoring_fcn: Callable[
            [atraj.TrajectoriesData, atraj.TrajectoriesData], Tuple[Metric, float]
        ],
        test_group="Test",
    ) -> PredictionResult:
        """runs a scoring function from a prediction configuration"""

        data_traj = Benchmark(prediction.benchmark).load_trajectories()

        pred_traj = Benchmark(
            BenchmarkConfiguration(
                name=prediction.benchmark.name,
                data_csv=prediction.pred_csv,
                prediction_dir=prediction.benchmark.prediction_dir,
                states=prediction.benchmark.states,
                groups=[test_group],
                time=prediction.benchmark.time,
                traj=prediction.benchmark.traj,
            )
        ).load_trajectories()

        m, v = scoring_fcn(data_traj[test_group], pred_traj[test_group])

        return PredictionResult(
            model_name=prediction.model_name,
            benchmark_name=prediction.benchmark.name,
            metric=m,
            value=v,
        )
