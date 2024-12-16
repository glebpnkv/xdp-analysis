import h5py
import numpy as np
import pandas as pd
import re


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

    df_out["experiment"] = file.split("/")[-1].split(".")[0]
    df_out["subject"] = re.search(r"Mobility_Lab_Subject_Export_([A-Z0-9]+)_\d{8}-\d{6}", file).group(1)

    return df_out
