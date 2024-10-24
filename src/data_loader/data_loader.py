from zipfile import ZipFile
import io
import os
import requests

import pandas as pd

class HARDataLoader:
    """
    class HARDataLoader:
        """
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

    def __init__(self):
        self.df_train_x: pd.DataFrame | None = None
        self.df_train_x_tr: pd.DataFrame | None = None
        self.df_train_y: pd.DataFrame | None = None

        self.df_test_x: pd.DataFrame | None = None
        self.df_test_x_tr: pd.DataFrame | None = None
        self.df_test_y: pd.DataFrame | None = None

    def load(self) -> None:
        """Loads Human Activity Recognition (HAR) dataset from zip file."""
        # Loading the zip file with the dataset
        response = requests.get(self.url)
        file_bytes = io.BytesIO(response.content)

        # Traversing through the dataset's structure
        data_file = ZipFile(file_bytes, "r")
        data_file = ZipFile(
            io.BytesIO(data_file.read("UCI HAR Dataset.zip")), "r"
        )

        # Getting activity labels
        df_activity_names = pd.read_csv(
            io.BytesIO(data_file.read("UCI HAR Dataset/activity_labels.txt")),
            header=None,
            sep='\s+',
            names=["activity", "name"]
        )

        # Extracting train data
        self.df_train_x_tr = self._make_features_tr(data_file, "train")
        self.df_train_y = self._make_targets(data_file, "train")

        # Extracting test data
        self.df_test_x_tr = self._make_features_tr(data_file, "test")
        self.df_test_y = self._make_targets(data_file, "test")

    def _make_features_tr(self,
                          data_file: ZipFile,
                          subset: str) -> pd.DataFrame:
        """Extracts DataFrame of targets from zip file."""
        cur_path_names = os.path.join(
            "UCI HAR Dataset",
            f"features.txt"
        )
        df_names = pd.read_csv(
            io.BytesIO(data_file.read(cur_path_names)),
            header=None,
            sep='\s+',
            names=["id", "name"]
        )

        cur_path = os.path.join(
            "UCI HAR Dataset",
            subset,
            f"X_{subset}.txt"
        )

        df_out = pd.read_csv(
            io.BytesIO(data_file.read(cur_path)),
            header=None,
            sep='\s+',
        )

        df_out.rename(columns=df_names["name"].to_dict(), inplace=True)

        return df_out

    def _make_targets(self,
                      data_file: ZipFile,
                      subset: str) -> pd.DataFrame:
        """Extracts DataFrame of targets from zip file."""
        cur_path = os.path.join(
            "UCI HAR Dataset",
            subset,
            f"y_{subset}.txt"
        )

        df_out = pd.read_csv(
            io.BytesIO(data_file.read(cur_path)),
            header=None,
            sep='\s+',
            names=["label"]
        )

        return df_out
