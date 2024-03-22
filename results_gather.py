#!/usr/bin/env python

import json
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_macro_f1(confusion_matrix):
    num_classes = confusion_matrix.shape[0]

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Adding a small epsilon to avoid division by zero

    macro_f1 = np.mean(f1_scores)
    return macro_f1


dataset_dirs = Path("./experiments/HyperparamSelect").iterdir()
for ds in sorted(dataset_dirs, key=lambda x: x.name.lower()):
    if not ds.is_dir():
        continue
    print(f"################ {ds.name} ################\n")
    curr_ds_accu = pd.DataFrame()
    curr_ds_f1 = pd.DataFrame()
    for rec_dir in ds.iterdir():
        if not rec_dir.is_dir():
            continue
        reception = rec_dir.name.split("-")[1]
        for proto_len_dir in rec_dir.iterdir():
            proto_len = proto_len_dir.name.split("-")[1]
            f1_to_avg = []
            accu_to_avg = []
            for fold_dir in proto_len_dir.iterdir():
                try:
                    conf_mat = np.loadtxt(fold_dir / "confusion_matrix.txt")
                    f1 = calculate_macro_f1(conf_mat)
                    with open(fold_dir / "results.json") as f:
                        accu = json.load(f)["accu_test"]
                    f1_to_avg.append(f1)
                    accu_to_avg.append(accu)
                except FileNotFoundError as e:
                    print(f"Skipping {fold_dir} due to missing files: {e}")
            if accu_to_avg:
                curr_ds_accu.loc[proto_len, reception] = np.mean(accu_to_avg)
                curr_ds_f1.loc[proto_len, reception] = np.mean(f1_to_avg)
            else:
                curr_ds_accu.loc[proto_len, reception] = np.nan
                curr_ds_f1.loc[proto_len, reception] = np.nan
    if curr_ds_accu.empty:
        continue
    formatted_dfs = []

    for df, kind in [(curr_ds_accu, "Accu"), (curr_ds_f1, "F1")]:
        df.sort_index(key=lambda x: x.astype(float), inplace=True)
        df = df.loc[:, sorted(df.columns, key=float)]

        top_3_values = sorted(df.values.flatten(), reverse=True)[:3]

        def format_value(val):
            if val in top_3_values:
                return f"\033[3;35m{val:.3f}\033[0m"  # Bold ANSI code
            return f"\033[0;00m{val:.3f}\033[0m"

        formatted_df = df.applymap(format_value)
        formatted_df.columns = [f"\033[0;00m{col}\033[0m" for col in formatted_df.columns]

        formatted_dfs.append(formatted_df)

        # Display the DataFrame with bolded values in the terminal
        print(kind)
        print(formatted_df.to_string(justify="right"))
        print()
