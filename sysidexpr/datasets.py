"""Data Access for S3 Hosted Datasets"""
import boto3
import botocore
import pandas as pd
import sys
import os

if sys.version_info[0] < 3:
    from StringIO import StringIO  # Python 2.x
else:
    from io import StringIO  # Python 3.x


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
            "default_region": "sfo3",
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


class GothPlasmaRaw(S3CsvLoader):
    """Do not use"""

    object_key = "Data/Plasma data/Raw/Goth_Plasma_WRAP_dem_pacc_082022.csv"


class PibImagingAnnotated(S3CsvLoader):
    object_key = "Data/PIB imaging data/Annotated/pib_roi_annotated.csv"


class PibImagingRaw(S3CsvLoader):
    object_key = "Data/Imaging data/Raw/pib_roi.csv"
