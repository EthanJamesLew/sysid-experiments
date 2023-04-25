"""constants for sysidexpr

NOTE: constants are made configurable to allow for easy use between google drive
and a local machine. This is an unusual design decision, but it is a convenient
way to allow for easy use of the package.

Maybe in the future, this should be changed to a namespace.
"""
import os
import pathlib

import pydantic
from pydantic import BaseModel

from sysidexpr.model import BenchmarkConfiguration

# name data base path
data_base_path = pathlib.Path("./drive/Shareddrives/Collaboration with Google/Data")
scaled_data_base_path = pathlib.Path(
    "./drive/Shareddrives/Collaboration with Google/ScaledData"
)


# name predictions base path
predictions_base_path = pathlib.Path(
    "./drive/Shareddrives/Collaboration with Google/Predictions"
)


# results path
scores_base_path = pathlib.Path(
    "./drive/Shareddrives/Collaboration with Google/Results/Scores"
)


def validate_consts():
    """validates the constants"""
    assert os.path.isdir(data_base_path), f"path {data_base_path} is not a directory"
    assert os.path.isdir(
        predictions_base_path
    ), f"path {predictions_base_path} is not a directory"
    assert os.path.isdir(
        scores_base_path
    ), f"path {scores_base_path} is not a directory"


class ConstConfig(BaseModel):
    """model to change the package constants

    this is an unusal design decisions, as it will change the global
    values specified here
    """

    data_base_path: pathlib.Path
    predictions_base_path: pathlib.Path
    scores_base_path: pathlib.Path

    # setup validators
    @pydantic.validator("data_base_path", pre=True)
    def data_base_path_exists(cls, v):
        if not os.path.isdir(v):
            raise ValueError(f"path {v} is not a directory")
        return v

    @pydantic.validator("predictions_base_path", pre=True)
    def predictions_base_path_exists(cls, v):
        if not os.path.isdir(v):
            raise ValueError(f"path {v} is not a directory")
        return v

    @pydantic.validator("scores_base_path", pre=True)
    def scores_base_path_exists(cls, v):
        if not os.path.isdir(v):
            raise ValueError(f"path {v} is not a directory")
        return v


def load_constants(config: ConstConfig):
    """loads the constants from the config"""
    global data_base_path
    global predictions_base_path
    global scores_base_path

    data_base_path = config.data_base_path
    predictions_base_path = config.predictions_base_path
    scores_base_path = config.scores_base_path

    # validate the constants
    validate_consts()
    update_configurations()


def load_constants_from_json(json_path: pathlib.Path):
    """loads the constants from a json file"""
    config = ConstConfig.parse_file(json_path)
    load_constants(config)


plasma_config = None
imaging_config = None
benchmarks = None


def update_configurations():
    global plasma_config
    global imaging_config
    global benchmarks

    # create a list of default prediction configurations
    plasma_config = BenchmarkConfiguration(
        name="plasma",
        data_csv=scaled_data_base_path
        / "plasma"
        / "Goth_Plasma_WRAP_dem_pacc_082022_annotated.csv",
        prediction_dir=predictions_base_path / "Plasma data",
        states=["ABeta_1_40", "ABeta_1_42", "pTau231", "pTau181", "GFAP", "NFL"],
        groups=["IsBaseline", "Train", "Validate", "Test"],
        time="AgeAtVisit",
        traj="WRAPNo",
    )

    imaging_config = BenchmarkConfiguration(
        name="imaging",
        data_csv=scaled_data_base_path / "imaging" / "pib_roi_annotated.csv",
        prediction_dir=predictions_base_path / "Imaging data",
        states=[
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
        ],
        groups=["IsBaseline", "Train", "Validate", "Test"],
        time="pib_age",
        traj="wrapno",
    )

    # create a list of benchmark configurations
    benchmarks = [plasma_config, imaging_config]


# default
update_configurations()
