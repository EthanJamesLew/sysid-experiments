"""Data Access for S3 Hosted Datasets"""
from typing import Dict
import boto3
import botocore
import pandas as pd
import sys
import os

if sys.version_info[0] < 3:
    from StringIO import StringIO  # Python 2.x
else:
    from io import StringIO  # Python 3.x

import autokoopman.core.trajectory as atraj


class S3CsvLoader:
    """Base class for loading CSVs from S3"""

    bucket_name = "sysidexpr"
    object_key = None

    @staticmethod
    def get_s3_client():
        # check that the environment variables are set
        for key in {"SYSIDEXPR_S3", "SYSIDEXPR_ACCESS_KEY", "SYSIDEXPR_SECRET_KEY"}:
            if key not in os.environ:
                raise ValueError(f"Environment variable {key} is not set")

        # these are secrets, so don't put them here
        # setup a .env file in the root of the project
        s3_configs = {
            "default_region": os.environ.get("SYSIDEXPR_REGION"),
            "default_endpoint": os.environ.get("SYSIDEXPR_S3"),
            "bucket_access_key": os.environ.get("SYSIDEXPR_ACCESS_KEY"),
            "bucket_secret_key": os.environ.get("SYSIDEXPR_SECRET_KEY"),
        }

        return boto3.client(
            "s3",
            config=botocore.config.Config(s3={"addressing_style": "virtual"}),
            region_name=s3_configs["default_region"],
            endpoint_url=s3_configs["default_endpoint"],
            aws_access_key_id=s3_configs["bucket_access_key"],
            aws_secret_access_key=s3_configs["bucket_secret_key"],
        )

    @classmethod
    def load_csv(cls) -> pd.DataFrame:
        """Load the CSV from S3"""
        client = cls.get_s3_client()
        csv_obj = client.get_object(Bucket=cls.bucket_name, Key=cls.object_key)
        body = csv_obj["Body"]
        csv_string = body.read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_string))
        return df

    def __init__(self) -> None:
        self.df = self.load_csv()


class AnnotatedCsvLoader(S3CsvLoader):
    """Base class for loading annotated CSVs"""

    @property
    def descr(self):
        """add the README table here"""
        raise NotImplementedError

    @property
    def state_names(self):
        raise NotImplementedError

    @property
    def time_name(self):
        raise NotImplementedError

    @property
    def subject_id_name(self):
        raise NotImplementedError

    def load_trajectories(self) -> Dict[str, atraj.TrajectoriesData]:
        """Load the trajectories from the CSV"""

        def extract_time_states(df):
            times = s_df[self.time_name].to_numpy()
            states = s_df[self.state_names].to_numpy()
            return atraj.Trajectory(
                times=times, states=states, inputs=None, state_names=self.state_names
            )

        # group by subject id
        subjects_dfs = self.df.groupby(self.df[self.subject_id_name])

        baseline_trajs = {}
        train_trajs = {}
        validate_trajs = {}
        test_trajs = {}

        for sid, s_df in subjects_dfs:
            baseline_t = extract_time_states(s_df["IsBaseline"])
            train_t = extract_time_states(s_df["Train"])
            validate_t = extract_time_states(s_df["Validate"])
            test_t = extract_time_states(s_df["Test"])

            baseline_trajs[sid] = baseline_t
            train_trajs[sid] = train_t
            validate_trajs[sid] = validate_t
            test_trajs[sid] = test_t

        return {
            "IsBaseline": atraj.TrajectoriesData(baseline_trajs),
            "Train": atraj.TrajectoriesData(train_trajs),
            "Validate": atraj.TrajectoriesData(validate_trajs),
            "Test": atraj.TrajectoriesData(test_trajs),
        }


class GothPlasmaAnnotated(S3CsvLoader):
    """
    Goth Plasma Data

    1. Compute the vector field using the train and test data. Use the Validate data to tune
    the hyperparameters of the vector field.
    2. Use the baseline visit of the test set (initial condition) to generate predictions for the
    next visits in this set.
    """

    object_key = (
        "Data/Plasma data/Annotated/Goth_Plasma_WRAP_dem_pacc_082022_annotated.csv"
    )

    @property
    def descr(self):
        """add the README table here"""
        return {
            "WRAPNo": "Subject ID",
            "AgeAtVisit": "Age at Visit",
            "ABeta_1_40": "Biomarker 1",
            "ABeta_1_42": "Biomarker 2",
            "pTau231": "Biomarker 3",
            "pTau181": "Biomarker 4",
            "GFAP": "Biomarker 5",
            "NFL": "Biomarker 6",
            "IsBaseline": "Indicator (T,F) for the baseline visit",
            "Train": "Indicator (T,F) for the training set",
            "Validate": "Indicator (T,F) for the validation set",
            "Test": "Indicator (T,F) for the test set",
        }

    @property
    def state_names(self):
        return ["ABeta_1_40", "ABeta_1_42", "pTau231", "pTau181", "GFAP", "NFL"]

    @property
    def time_name(self):
        return "AgeAtVisit"

    @property
    def subject_id_name(self):
        return "WRAPNo"


class GothPlasmaRaw(S3CsvLoader):
    """Do not use"""

    object_key = "Data/Plasma data/Raw/Goth_Plasma_WRAP_dem_pacc_082022.csv"


class PibImagingAnnotated(S3CsvLoader):
    object_key = "Data/Imaging data/Annotated/pib_roi_annotated.csv"

    @property
    def state_names(self):
        return [
            "dvr_precentral_l",
            "dvr_precentral_r",
            "dvr_frontal_sup_l",
            "dvr_frontal_sup_r",
            "pet_date_mri_date_diff_days",
            "dvr_frontal_sup_orb_l",
            "dvr_frontal_sup_orb_r",
            "dvr_frontal_mid_l",
            "dvr_frontal_mid_r",
            "dvr_frontal_mid_orb_l",
            "dvr_frontal_mid_orb_r",
            "dvr_frontal_inf_oper_l",
            "dvr_frontal_inf_oper_r",
            "dvr_frontal_inf_tri_l",
            "dvr_frontal_inf_tri_r",
            "dvr_frontal_inf_orb_l",
            "dvr_frontal_inf_orb_r",
            "dvr_rolandic_oper_l",
            "dvr_rolandic_oper_r",
            "dvr_supp_motor_area_l",
            "dvr_supp_motor_area_r",
            "dvr_olfactory_l",
            "dvr_olfactory_r",
            "dvr_frontal_sup_medial_l",
            "dvr_frontal_sup_medial_r",
            "dvr_frontal_med_orb_l",
            "dvr_frontal_med_orb_r",
            "dvr_rectus_l",
            "dvr_rectus_r",
            "dvr_insula_l",
            "dvr_insula_r",
            "dvr_cingulum_ant_l",
            "dvr_cingulum_ant_r",
            "dvr_cingulum_mid_l",
            "dvr_cingulum_mid_r",
            "dvr_cingulum_post_l",
            "dvr_cingulum_post_r",
            "dvr_hippocampus_l",
            "dvr_hippocampus_r",
            "dvr_parahippocampal_l",
            "dvr_parahippocampal_r",
            "dvr_amygdala_l",
            "dvr_amygdala_r",
            "dvr_calcarine_l",
            "dvr_calcarine_r",
            "dvr_cuneus_l",
            "dvr_cuneus_r",
            "dvr_lingual_l",
            "dvr_lingual_r",
            "dvr_occipital_sup_l",
            "dvr_occipital_sup_r",
            "dvr_occipital_mid_l",
            "dvr_occipital_mid_r",
            "dvr_occipital_inf_l",
            "dvr_occipital_inf_r",
            "dvr_fusiform_l",
            "dvr_fusiform_r",
            "dvr_postcentral_l",
            "dvr_postcentral_r",
            "dvr_parietal_sup_l",
            "dvr_parietal_sup_r",
            "dvr_parietal_inf_l",
            "dvr_parietal_inf_r",
            "dvr_supramarginal_l",
            "dvr_supramarginal_r",
            "dvr_angular_l",
            "dvr_angular_r",
            "dvr_precuneus_l",
            "dvr_precuneus_r",
            "dvr_paracentral_lobule_l",
            "dvr_paracentral_lobule_r",
            "dvr_caudate_l",
            "dvr_caudate_r",
            "dvr_putamen_l",
            "dvr_putamen_r",
            "dvr_pallidum_l",
            "dvr_pallidum_r",
            "dvr_thalamus_l",
            "dvr_thalamus_r",
            "dvr_heschl_l",
            "dvr_heschl_r",
            "dvr_temporal_sup_l",
            "dvr_temporal_sup_r",
            "dvr_temporal_pole_sup_l",
            "dvr_temporal_pole_sup_r",
            "dvr_temporal_mid_l",
            "dvr_temporal_mid_r",
            "dvr_temporal_pole_mid_l",
            "dvr_temporal_pole_mid_r",
            "dvr_temporal_inf_l",
            "dvr_temporal_inf_r",
            "dvr_cerebelum_crus1_l",
            "dvr_cerebelum_crus1_r",
            "dvr_cerebelum_crus2_l",
            "dvr_cerebelum_crus2_r",
            "dvr_cerebelum_3_l",
            "dvr_cerebelum_3_r",
            "dvr_cerebelum_4_5_l",
            "dvr_cerebelum_4_5_r",
            "dvr_cerebelum_6_l",
            "dvr_cerebelum_6_r",
            "dvr_cerebelum_7b_l",
            "dvr_cerebelum_7b_r",
            "dvr_cerebelum_8_l",
            "dvr_cerebelum_8_r",
            "dvr_cerebelum_9_l",
            "dvr_cerebelum_9_r",
            "dvr_cerebelum_10_l",
            "dvr_cerebelum_10_r",
            "dvr_vermis_1_2",
            "dvr_vermis_3",
            "dvr_vermis_4_5",
            "dvr_vermis_6",
            "dvr_vermis_7",
            "dvr_vermis_8",
            "dvr_vermis_9",
            "dvr_vermis_10",
        ]

    @property
    def time_name(self):
        return "pib_age"

    @property
    def subject_id_name(self):
        return "wrapno"


class PibImagingRaw(S3CsvLoader):
    object_key = "Data/Imaging data/Raw/pib_roi.csv"
