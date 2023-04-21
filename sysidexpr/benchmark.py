import pandas as pd
import numpy as np
from sysidexpr.model import (
    BenchmarkConfiguration,
    Metric,
    PredictionConfiguration,
    PredictionResult,
)
from typing import Callable, Tuple
import autokoopman.core.trajectory as atraj


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
                trajectories[groupname][sid] = extract_time_states(
                    s_df.loc[s_df[groupname] == True]
                )

        return {g: atraj.TrajectoriesData(ts) for g, ts in trajectories.items()}

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
