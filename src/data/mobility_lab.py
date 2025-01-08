import os
import re
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ca_tcc.dataloader.augmentations import jitter, permutation
from models.mobility_lab import MobilityLabConfigs


summary_value_cols = ["Normative Mean", "Normative StDev", "Mean", "StDev"]


def dataset_to_df_long(ds, name):
    data = ds[()]

    rows = data.shape[0]
    cols = 1
    if len(data.shape) > 1:
        cols = data.shape[-1]

    reshaped_data = data.reshape(-1)

    df_out = pd.DataFrame({
        "step": np.repeat(np.arange(rows), cols),
        "channel": np.tile(np.arange(cols), rows),
        "value": reshaped_data
    })
    df_out["feature"] = name.lower()

    df_out = df_out[["step", "feature", "channel", "value"]]

    return df_out

def process_key(dset, key):
    df_out = pd.concat(
        [
            dataset_to_df_long(v, k)
            for k, v in dset.items()
            if isinstance(v, h5py.Dataset)
        ],
        ignore_index=True
    )
    df_out['key'] = key

    return df_out

def process_group(dset):
    out = []

    for k, v in dset.items():
        out.append(
            process_key(v, k)
        )

    return pd.concat(out, ignore_index=True)

def process_experiment_file(file):
    # Reading h5 file
    f = h5py.File(file, 'r')

    file_groups = [
        k for k, v in f.items()
        if isinstance(v, h5py.Group)
    ]

    if len(file_groups) == 0:
        return None

    out = []

    # Iterating over groups
    for cur_group in file_groups:
        out.append(
            process_group(f[cur_group])
        )

    df_out = pd.concat(out, ignore_index=True)

    # Reshaping back to wide format
    df_out = df_out.pivot(
        columns=['feature', 'channel'],
        index=['key', 'step'],
        values='value'
    )

    if df_out.empty:
        return None

    col_value_counts = df_out.columns.get_level_values(0).value_counts()

    df_out.columns = [
        "_".join((str(y) for y in x))
        if bool(col_value_counts[x[0]] > 1)
        else x[0]
        for x in df_out.columns
    ]
    df_out["time"] = pd.to_datetime(df_out['time'], unit='us')
    df_out = df_out.reset_index(level='step', drop=True).set_index('time', append=True).sort_index()

    df_out["raw_file_name"] = Path(file).name

    return df_out

def process_experiment_summary_file(file):
    df_out = pd.read_csv(
        file,
        skiprows=12,
        usecols=["Measure"] + summary_value_cols
    )

    # Fixing the formatting of numerical columns
    for col in summary_value_cols:
        if not pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = pd.to_numeric(df_out[col].str.replace(",", "."), errors="coerce")

    return df_out


class TrainingMode(Enum):
    self_supervised = "self_supervised"
    fine_tune = "fine_tune"
    SupCon = "SupCon"
    SimCLR = "SimCLR"


class MobilityLabDataset(Dataset):
    """
    Represents a dataset for MobilityLab experiments.

    This class is a custom Dataset implementation to interface with data used in
    Mobility Lab machine learning experiments. It provides access to preprocessed
    data stored in parquet files and corresponding target labels. The dataset
    assumes that raw data files are stored in the specified directory and targets
    are supplied via a DataFrame.

    Attributes:
        data_dir: str
            Path to the directory where raw data files are stored.
        df_targets: pandas.DataFrame
            DataFrame containing the mapping of file names to their respective
            target labels.

    Methods:
        __len__:
            Returns the total number of samples in the dataset.
        __getitem__:
            Retrieves the raw data and target at the specified index.
    """

    def __init__(
        self,
        data_dir: str,
        df_targets: pd.DataFrame,
        training_mode: TrainingMode = TrainingMode.self_supervised,
        max_segments: int = 10
    ) -> None:
        """
        Represents the initialization of an object with specified data directory and
        target data.

        Attributes:
        data_dir: str
            Directory path where relevant data is stored.
        df_targets: pandas.DataFrame
            DataFrame containing target information for processing.

        """
        self.data_dir = data_dir
        self.df_targets = df_targets
        self.training_mode = training_mode
        self.max_segments = max_segments

    def __len__(self):
        return len(self.df_targets)

    def __getitem__(self, idx):
        cur_file = self.df_targets.loc[idx, "raw_file_name"]
        df_raw = pd.read_parquet(
            os.path.join(self.data_dir, f'raw_file_name={cur_file}'),
        )

        x = torch.from_numpy(
            np.stack(
                df_raw.groupby("key").apply(lambda z: z.values)
            )
        )

        y = torch.tensor(self.df_targets.loc[idx, "target"])

        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            # Weak augmentation - swapping channels
            df_raw_aug = df_raw.groupby("key").apply(lambda z: z.values)
            df_raw_aug[['11948', '11966']] = df_raw_aug[['11966', '11948']].values
            weak_aug = torch.from_numpy(
                np.stack(
                    df_raw_aug
                )
            )

            # Strong augmentation - jitter and permutation
            strong_aug = jitter(
                permutation(
                    x,
                    max_segments=self.max_segments
                )
            )
            return x, y, weak_aug, strong_aug
        else:
            return x, y, x, x


class MobilityLabDataController:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @staticmethod
    def _collate_to_tensor(tensors):
        if tensors[0].dim() > 0:
            # Convert to nested tensor and pad
            out = torch.nested.to_padded_tensor(
                input=torch.nested.nested_tensor(list(tensors)),
                padding=0.0
            ).type(torch.float32)
        else:
            # Handle 0-dimensional tensors with labels
            out = torch.Tensor(tensors).type(tensors[0].dtype)

        return out

    @staticmethod
    def _collate_fn(batch):
        return tuple(
            map(MobilityLabDataController._collate_to_tensor, zip(*batch))
        )

    def get_torch_dataset(
        self,
        df_targets: pd.DataFrame,
        config: MobilityLabConfigs,
        training_mode: TrainingMode = TrainingMode.self_supervised,
        shuffle: bool = False,
    ):
        dataset = MobilityLabDataset(
            data_dir=self.data_dir,
            df_targets=df_targets,
            training_mode=training_mode
        )

        dl = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

        return dl
