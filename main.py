#!/usr/bin/env python

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torcheval.metrics.functional import multiclass_confusion_matrix

from artificial_protos_datasets import ArtificialProtos
from autoencoder import PermutingConvAutoencoder, RegularConvAutoencoder, VariationalConvAutoencoder, train_autoencoder, train_vae_with_anchors
from datasets_utils import TSCDataset, ds_load, transform_ts_data
from model import ProtoTSNet
from train import EpochType, ProtoTSCoeffs, create_logger, train_prototsnet

device = torch.device("cuda")
# torch.cuda.set_per_process_memory_fraction(fraction=0.5, device=0)  # or 1, watch out for CUDA_VISIBLE_DEVICES

# dataset in arff format should be put in the 'datasets/' directory (downloaded from timeseriesclassification.com)
DATASETS_PATH = Path("datasets")


# write your own function to load your dataset...
def custom_dataset():
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    X_test = None
    y_test = None
    train_ds = TSCDataset(X_train, y_train)
    val_ds = TSCDataset(X_val, y_val) if X_val is not None else None
    test_ds = TSCDataset(X_test, y_test)
    if val_ds is not None:
        return train_ds, val_ds, test_ds
    return train_ds, test_ds


# ... like this:
# and update the code below to use it (look for pm_dataset usage)
def pm_dataset(use_pm=True):
    # replicates the transformations described in the LAXCAT paper (to the best of our understanding)
    df = pd.read_csv("./datasets/PRSA_data_2010.1.1-2014.12.31.csv")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.set_index("datetime", inplace=True)
    df = df.dropna()

    day_counts = df.groupby(df.index.date).size()
    full_days = day_counts[day_counts == 24].index
    df = df[pd.Index(df.index.date).isin(full_days)]
    wd = {
        "cv": 0,
        "NW": 1,
        "NE": 2,
        "SE": 3,
    }
    df["cbwd"] = df["cbwd"].map(wd)
    dates = df.index.date
    unique_dates = set(dates)

    xs = []
    ys = []
    for date in unique_dates:
        next_day = date + pd.Timedelta(days=1)
        if next_day in unique_dates and next_day.weekday() < 5:
            next_day_8am = pd.Timestamp(next_day) + pd.Timedelta(hours=8)
            data = df.loc[df.index.date == date, ["pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]].to_numpy()
            if not use_pm:
                data = data[:, 1:]
            pm_value = df.loc[next_day_8am, "pm2.5"]
            xs.append(data)
            # quantize pm2.5 according to US EPA standards
            ys.append(
                0 if pm_value < 12 else 1 if pm_value < 35.4 else 2 if pm_value < 55.4 else 3 if pm_value < 150.4 else 4 if pm_value < 250.4 else 5
            )

    X = np.array(xs, dtype=np.float32).transpose(0, 2, 1)
    y = np.array(ys)
    X_train = X[: int(0.7 * X.shape[0])]
    y_train = y[: int(0.7 * X.shape[0])]
    X_test = X[int(0.7 * X.shape[0]) :]
    y_test = y[int(0.7 * X.shape[0]) :]
    train_ds = TSCDataset(X_train, y_train)
    test_ds = TSCDataset(X_test, y_test)
    return train_ds, test_ds


DATA_PATH = Path("/mnt/z/dane/fin-ta/")
INTERVAL = "1d"
SLICE_LEN = 80
AFTER_RIGHT_SHOULDER = 7


def read_ohlc_with_ta_and_trend(symbol):
    full_data_path = DATA_PATH / INTERVAL
    ohlc_df = pd.read_parquet(full_data_path / f"{symbol}.parquet")
    ta_df = pd.read_parquet(full_data_path / f"{symbol}-ta2.parquet")
    trend_df = pd.read_parquet(full_data_path / f"{symbol}-trends2.parquet")
    return ohlc_df.join(ta_df).join(trend_df)


def formation_end_idxs(df, formation):
    return np.nonzero((df[formation].iloc[:, [-1]] != 0).any(axis=1))[0]


def get_slice_ending_with(series, end_idx, length):
    return series.iloc[max(0, end_idx - length + 1) : end_idx + 1].copy()


def ta_dataset(get_formation_labels=False):

    symbols = [f.stem[:-4] for f in (DATA_PATH / INTERVAL).glob("*-ta2.parquet")]
    ticker_dfs = {}
    for symbol in symbols:
        ticker_dfs[symbol] = read_ohlc_with_ta_and_trend(symbol)
    c0 = []
    formation_labels_X0 = []
    for instr in symbols:
        df = ticker_dfs[instr]
        idxs = formation_end_idxs(df, "UPTREND")
        idxs = np.concatenate([idxs, formation_end_idxs(df, "DOWNTREND")])
        for idx in idxs:
            slice = get_slice_ending_with(df[(instr, "Close")], idx, SLICE_LEN)
            if slice.shape[0] < SLICE_LEN:
                continue
            if df["UPTREND"].iloc[idx, -1] != 0:
                formation_labels_X0.append(0)
            elif df["DOWNTREND"].iloc[idx, -1] != 0:
                formation_labels_X0.append(1)
            else:
                raise ValueError(f"Unknown formation type for {instr} at index {idx}")
            c0.append(slice)
    X_0 = np.stack(c0)

    c1 = []
    formation_labels_X1 = []
    for instr in symbols:
        df = ticker_dfs[instr]
        idxs = formation_end_idxs(df, "HS")
        idxs = np.concatenate([idxs, formation_end_idxs(df, "IHS")])
        for idx in idxs:
            slice = get_slice_ending_with(df[(instr, "Close")], idx + AFTER_RIGHT_SHOULDER, SLICE_LEN)
            if slice.shape[0] < SLICE_LEN:
                continue
            if df["HS"].iloc[idx, -1] != 0:
                formation_labels_X1.append(2)
            elif df["IHS"].iloc[idx, -1] != 0:
                formation_labels_X1.append(3)
            else:
                raise ValueError(f"Unknown formation type for {instr} at index {idx}")
            c1.append(slice)
    X_1 = np.stack(c1)

    np.random.seed(42)
    perm_0 = np.random.permutation(X_0.shape[0])
    perm_1 = np.random.permutation(X_1.shape[0])
    X_0 = X_0[perm_0]
    X_1 = X_1[perm_1]
    formation_labels_X0 = np.array(formation_labels_X0)[perm_0]
    formation_labels_X1 = np.array(formation_labels_X1)[perm_1]
    assert X_0.shape[0] == len(formation_labels_X0)
    assert X_1.shape[0] == len(formation_labels_X1)

    # sample equal number of samples from each class
    smaller_class = min(X_0.shape[0], X_1.shape[0])
    print(f'Smaller class is class {"monotonic" if X_0.shape[0] < X_1.shape[0] else "HS"}: {X_0.shape[0]} vs {X_1.shape[0]}')
    X_0 = X_0[:smaller_class]
    X_1 = X_1[:smaller_class]
    formation_labels_X0 = formation_labels_X0[:smaller_class]
    formation_labels_X1 = formation_labels_X1[:smaller_class]

    X = np.concatenate([X_0, X_1])
    y = np.concatenate([np.zeros(X_0.shape[0]), np.ones(X_1.shape[0])])
    formation_labels = np.concatenate([formation_labels_X0, formation_labels_X1])

    # permute data
    np.random.seed(42)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    formation_labels = formation_labels[perm]

    # normalize each series separately
    X -= X.mean(axis=1)[:, np.newaxis]
    X /= X.std(axis=1)[:, np.newaxis]

    X = X.astype(np.float32).reshape(X.shape[0], 1, X.shape[1])
    y = np.array(y).astype(np.int64)
    X_train = X[: int(0.7 * X.shape[0])]
    y_train = y[: int(0.7 * X.shape[0])]
    X_test = X[int(0.7 * X.shape[0]) :]
    y_test = y[int(0.7 * X.shape[0]) :]
    train_ds = TSCDataset(X_train, y_train)
    test_ds = TSCDataset(X_test, y_test)

    if get_formation_labels:
        formation_labels_train = formation_labels[: int(0.7 * X.shape[0])]
        formation_labels_test = formation_labels[int(0.7 * X.shape[0]) :]
        return train_ds, test_ds, formation_labels_train, formation_labels_test

    return train_ds, test_ds


def artificial_dataset():
    log(f"Preparing artificial dataset...", flush=True, display=True)
    train_art = ArtificialProtos(1000, feature_noise_power=0.1, randomize_right_side=True)
    train_ds = TSCDataset(train_art.X, train_art.y)
    test_art = ArtificialProtos(100, feature_noise_power=0.1, randomize_right_side=True)
    test_ds = TSCDataset(test_art.X, test_art.y)
    return train_ds, test_ds


def experiment_setup(experiment_subpath, expert_protos_dir=None):
    # create subdirectory for the experiment and copy files to save state of the code
    experiment_dir = Path.cwd() / "experiments" / experiment_subpath
    os.makedirs(experiment_dir, exist_ok=True)

    shutil.copy(src=Path.cwd() / "autoencoder.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "datasets_utils.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "main.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "model.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "push.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "receptive_field.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "train.py", dst=experiment_dir)
    if expert_protos_dir is not None:
        shutil.copytree(src=expert_protos_dir, dst=experiment_dir / "target_protos", dirs_exist_ok=True)

    return experiment_dir


def validate_proto_len(proto_len):
    flen = float(proto_len)
    if flen < 0 or flen > 1:
        parser.error("proto_len must be in range (0, 1]")
    return flen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with specified dataset and experiment directory.")
    parser.add_argument(
        "--uea_dataset", type=str, help="Use this if you want to run experiment on UEA dataset under datasets/ directory", required=False
    )
    parser.add_argument("--artificial_dataset", action="store_true", help="Run experiments on artificial dataset", required=False)
    parser.add_argument("--experiment_name", type=str, required=True, help="Directory to save experiment results")
    parser.add_argument("--no_permuting_encoder", action="store_true", help="Do not use permuting encoder", required=False)
    parser.add_argument("--vae_encoder", action="store_true", help="Use VAE encoder, implies --no_permuting_encoder", required=False)
    parser.add_argument(
        "--vae_pretrain_beta",
        type=float,
        default=1.0,
        help="Beta used in VAE pretraining, only effective with VAE encoder and pretraining",
        required=False,
    )
    parser.add_argument(
        "--vae_pretrain_gamma",
        type=float,
        default=0.5,
        help="Gamma used in VAE pretraining (for target anchoring), only effective with VAE encoder and pretraining",
        required=False,
    )
    parser.add_argument(
        "--vae_anchor_warmup_epochs",
        type=int,
        default=20,
        help="Number of epochs to train VAE before target anchoring, only effective with VAE encoder and pretraining",
        required=False,
    )
    parser.add_argument("--no_encoder_pretraining", action="store_true", help="Do not pretrain encoder before ProtoTSNet training", required=False)
    parser.add_argument("--pretraining_epochs", type=int, help="Number of encoder pretraining epochs", required=False, default=50)
    parser.add_argument("--warm_epochs", type=int, help="Number of warm-up epochs", required=False, default=None)
    parser.add_argument("--push_start_epoch", type=int, help="Epoch to start pushing prototypes", required=False, default=110)
    parser.add_argument("--push_epochs_interval", type=int, help="Interval between pushing prototypes", required=False, default=30)
    parser.add_argument("--last_layer_epochs", type=int, help="Number of epochs to train last layer", required=False, default=40)
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", required=False, default=200)
    parser.add_argument("--proto_features", type=int, help="Number of latent features", required=False, default=32)
    parser.add_argument(
        "--proto_len", type=validate_proto_len, help="Prototype length (0-1 range, fraction of series length)", required=False, default=None
    )
    parser.add_argument("--protos_per_class", type=int, help="Number of prototypes for each class", required=False, default=10)
    parser.add_argument("--reception", type=float, help="Fraction of significant features", required=False, default=None)
    parser.add_argument(
        "--l1_addon_coeff", type=float, help="L1 regularization coefficient for feature importance layer", required=False, default=1e-3
    )
    parser.add_argument("--l1_coeff", type=float, help="L1 regularization coefficient", required=False, default=1e-3)
    parser.add_argument("--l2_target_protos_coeff", type=float, help="L2 regularization coefficient for target protos", required=False, default=1e-3)
    parser.add_argument("--clst_coeff", type=float, help="Cluster separation coefficient", required=False, default=0.08)
    parser.add_argument("--sep_coeff", type=float, help="Separation coefficient", required=False, default=-0.0001)
    parser.add_argument("--param_selection", action="store_true", help="Run hyperparameter selection", required=False, default=False)
    parser.add_argument("--verbose", action="store_true", help="Verbose output", required=False, default=False)
    parser.add_argument("--skip_scaling", action="store_true", help="Skip scaling of the dataset", required=False, default=False)
    parser.add_argument(
        "--penalize_class_0",
        action="store_true",
        help="Penalize class 0 positive connections in the last layer - if class 0 contains no prototypes, other classes are encouraged to have more meaningful ones",
        required=False,
        default=False,
    )

    parser.add_argument("--target_protos_dir", help="Directory with target protos", required=False, default=None)
    parser.add_argument(
        "--force_expert_protos",
        action="store_true",
        help="Force prototypes to be the expert ones, no prototype layer learning is done then",
        required=False,
        default=False,
    )

    args = parser.parse_args()

    uea_dataset = args.uea_dataset

    if uea_dataset is not None:
        # read best_params.csv
        try:
            best_params = pd.read_csv("best_params.csv", index_col=0)
        except Exception as e:
            best_params = None
    else:
        best_params = None

    if best_params is None:
        if not args.proto_len:
            parser.error("Prototype length must be provided for non-UEA datasets or when best_params.csv is missing")

    experiment_name = f"{args.experiment_name}/{uea_dataset}" if uea_dataset is not None else args.experiment_name
    experiment_dir = experiment_setup(experiment_name, args.target_protos_dir)
    log, logclose = create_logger(experiment_dir / "log.txt", display=args.verbose)

    if args.skip_scaling:
        skip_scaling = True
    elif best_params is not None:
        skip_scaling = best_params.loc[uea_dataset, "skip_scaling"] == "T"
    else:
        skip_scaling = False

    if uea_dataset is not None:
        log(f"Loading UEA dataset {uea_dataset}...", flush=True, display=True)
        train_ds, test_ds = ds_load(DATASETS_PATH, uea_dataset)
    elif args.artificial_dataset:
        train_ds, test_ds = artificial_dataset()
    else:
        log(f"Running on custom dataset, loading...", flush=True, display=True)
        train_ds, test_ds = ta_dataset()

    if not skip_scaling:
        log(f"Scaling dataset...", flush=True, display=True)
        scaler = StandardScaler()
        train_ds.X = transform_ts_data(train_ds.X, scaler)
        test_ds.X = transform_ts_data(test_ds.X, scaler, fit=False)
    else:
        log(f"Skipping scaling of the dataset", flush=True, display=True)

    # hyperparameters

    # number of prototypes will equal 'protos_per_class * number of classes'
    protos_per_class = args.protos_per_class

    # prototype length (number of time steps) - it is latent space length, so due to receptive field in the input space it is longer
    proto_len = float(best_params.loc[uea_dataset, "proto_len"]) if args.proto_len is None else args.proto_len

    # number of latent features (dimensions) that input is encoded to
    proto_features = args.proto_features

    # estimate for the fraction of significant features, better to underestimate than overestimate
    reception = float(best_params.loc[uea_dataset, "reception"]) if args.reception is None and best_params is not None else args.reception
    permuting_encoder = not args.no_permuting_encoder
    vae_encoder = args.vae_encoder
    if vae_encoder:
        permuting_encoder = False
    encoder_pretraining = not args.no_encoder_pretraining

    # number of epochs during which encoder weights are frozen, value >0 only makes sense if encoder is pretrained
    warm_epochs = args.warm_epochs if args.warm_epochs is not None else 50 if encoder_pretraining else 0

    # when to start pushing prototypes onto the input data
    push_start_epoch = args.push_start_epoch

    # which epochs to push prototypes on
    push_epochs = range(push_start_epoch, 1000, args.push_epochs_interval)

    # how many epochs to train the last layer (prototypes <-> class mapping)
    num_last_layer_epochs = args.last_layer_epochs

    # overall number of epochs (PUSH + last layer "epochs" count as one epoch here), set it so that the training ends with PUSH
    epochs = args.epochs

    # how much each element contributes to the loss, l1 is last layer l1 regularization, l1_addon is regularization of feature importance layer
    coeffs = ProtoTSCoeffs(
        crs_ent=1,
        l1_addon=args.l1_addon_coeff,
        l1=args.l1_coeff,
        clst=args.clst_coeff,
        sep=args.sep_coeff,
        l2_target_protos=args.l2_target_protos_coeff,
    )

    # whether to penalize class 0 connections - if class 0 contains no prototypes, we encourage other classes to have meaningful prototypes
    penalize_class_0 = args.penalize_class_0

    # retrieve details of the dataset
    num_classes = len(np.unique(train_ds.y))
    num_features = train_ds.X.shape[1]
    ts_len = train_ds.X.shape[2]

    if isinstance(proto_len, float):
        proto_len = int(ts_len * proto_len)

    if num_features == 1:
        # this is an univariate dataset, no point in using permuting encoder
        permuting_encoder = False
    elif not reception and permuting_encoder:
        parser.error("Reception must be provided for non-UEA datasets when using permuting encoder")

    if args.target_protos_dir is None and args.force_expert_protos:
        parser.error("Target protos directory must be provided when using --force_expert_protos")

    # TODO: ugly hardcoded value
    rf_size = proto_len + 12

    target_protos = {}
    if args.target_protos_dir is not None:
        target_protos_dir = Path(args.target_protos_dir)
        if not target_protos_dir.exists():
            parser.error(f"Target protos directory {target_protos_dir} does not exist")

        for class_idx in range(num_classes):
            target_protos[class_idx] = {}
            for proto_idx in range(args.protos_per_class):
                filepath = target_protos_dir / f"class_{class_idx}_proto_{proto_idx}.txt"
                if not filepath.exists():
                    break
                target_proto = np.loadtxt(filepath)
                if len(target_proto.shape) == 1:
                    target_proto = target_proto.reshape(1, -1)
                # target_proto -= target_proto.mean(axis=1)[:, np.newaxis]
                # target_proto /= target_proto.std(axis=1)[:, np.newaxis]
                # if not skip_scaling:
                #     target_proto = scaler.transform(target_proto.transpose(1, 0)).transpose(1, 0)
                target_protos[class_idx][proto_idx] = target_proto

            print(f"Class {class_idx}: {len(target_protos[class_idx])} target proto(s)")

        for class_idx in range(num_classes):
            assert (
                len(target_protos[class_idx]) <= args.protos_per_class
            ), f"Number of protos for class {class_idx} too large, expected at most {args.protos_per_class} prototypes, got {len(target_protos[class_idx])}"
            for proto_idx in range(len(target_protos[class_idx])):
                assert (
                    target_protos[class_idx][proto_idx].shape[0] == num_features
                ), f"Number of features in target proto {proto_idx} for class {class_idx} is incorrect, expected {num_features} feature(s), got {target_protos[class_idx][proto_idx].shape[0]}"
                assert (
                    target_protos[class_idx][proto_idx].shape[1] == rf_size
                ), f"Length of target proto {proto_idx} for class {class_idx} is incorrect, expected {rf_size} time steps, got {target_protos[class_idx][proto_idx].shape[1]}"

    def setup_and_run_experiment(experiment_name, experiment_dir, log, train_ds, test_ds, reception, proto_len):
        try:
            train_batch_size = 32
            # reduce in case dataset is small
            while train_batch_size > len(train_ds.X) / 4:
                train_batch_size //= 2
            test_batch_size = 128

            do_batch_norm = True
            if train_batch_size < 8:
                do_batch_norm = False

            whole_training_start = time.time()

            if permuting_encoder:
                autoencoder = PermutingConvAutoencoder(
                    num_features=num_features,
                    latent_features=proto_features,
                    reception_percent=reception,
                    padding="same",
                    do_max_pool=False,
                    do_batch_norm=do_batch_norm,
                )
            elif vae_encoder:
                autoencoder = VariationalConvAutoencoder(
                    num_features=num_features,
                    latent_features=proto_features,
                    padding="same",
                    do_max_pool=False,
                    do_batch_norm=do_batch_norm,
                    num_conv_filters=8,
                )
            else:
                autoencoder = RegularConvAutoencoder(
                    num_features=num_features,
                    latent_features=proto_features,
                    padding="same",
                    do_max_pool=False,
                    do_batch_norm=do_batch_norm,
                    num_conv_filters=8,
                )
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
            if encoder_pretraining:
                log(f"Training encoder", flush=True, display=True)
                # if vae_encoder:
                #     train_vae_with_anchors(
                #         autoencoder,
                #         train_loader,
                #         test_loader,
                #         target_protos,
                #         device=device,
                #         log=log,
                #         num_epochs=args.pretraining_epochs,
                #         beta=args.vae_pretrain_beta,
                #         gamma=args.vae_pretrain_gamma,
                #         anchor_warmup=args.vae_anchor_warmup_epochs,
                #     )
                # else:
                train_autoencoder(autoencoder, train_loader, test_loader, device=device, log=log, num_epochs=args.pretraining_epochs)
            # autoencoder.encoder.set_return_indices(False)

            ptsnet = ProtoTSNet(
                cnn_base=autoencoder.encoder,
                num_features=num_features,
                ts_sample_len=ts_len,
                proto_num=protos_per_class * num_classes,
                latent_features=proto_features,
                proto_len_latent=proto_len,
                num_classes=num_classes,
                prototype_activation_function="log",
                target_protos=target_protos,
            )

            def lr_sched_setup(optimizer, epoch_type):
                if epoch_type == EpochType.JOINT:
                    return torch.optim.lr_scheduler.CyclicLR(
                        optimizer, base_lr=1e-4, max_lr=3e-2, step_size_up=10, step_size_down=20, mode="exp_range", gamma=0.99, cycle_momentum=False
                    )
                elif epoch_type == EpochType.LAST_LAYER:
                    return torch.optim.lr_scheduler.CyclicLR(
                        optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=15, step_size_down=25, mode="exp_range", gamma=0.99, cycle_momentum=False
                    )
                return None

            log(f"Training ProtoTSNet", flush=True, display=True)
            trainer = train_prototsnet(
                ptsnet=ptsnet,
                experiment_dir=experiment_dir,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                num_warm_epochs=warm_epochs,
                push_start_epoch=push_start_epoch,
                push_epochs=push_epochs,
                num_last_layer_epochs=num_last_layer_epochs,
                num_epochs=epochs,
                coeffs=coeffs,
                lr_sched_setup=lr_sched_setup,
                log=log,
                penalize_class_0=args.penalize_class_0,
                force_expert_protos=args.force_expert_protos,
                add_params_to_log={
                    "encoder_pretraining": encoder_pretraining,
                    "permuting_encoder": permuting_encoder,
                    "reception": reception,
                    "proto_len_fraction": f"{proto_len / ts_len:.3f}",
                },
            )

            accu_test = trainer.latest_stat("accu_test")
            total_time = trainer.latest_stat("total_time")
            log(f"Last epoch test accu: {accu_test*100:.2f}%", display=True)
            with open(experiment_dir / "results.json", "w") as f:
                json.dump({"accu_test": accu_test, "time": total_time}, f, indent=4)

            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
            confusion_matrix = torch.zeros(ptsnet.num_classes, ptsnet.num_classes)
            for i, (image, label) in enumerate(test_loader):
                output, _ = ptsnet(image.to(device))
                confusion_matrix += multiclass_confusion_matrix(output.to("cpu"), label, num_classes=output.shape[1])
            np.savetxt(experiment_dir / "confusion_matrix.txt", confusion_matrix.numpy(), fmt="%4d")

            whole_training_end = time.time()
            log(f"Done in {trainer.curr_epoch - 1} epochs, {whole_training_end - whole_training_start:.2f}s", display=True)
        except Exception as e:
            log(f"Exception ocurred for experiment {experiment_name}: {e}", display=True)
            tb_str = traceback.format_tb(e.__traceback__)
            log("\n".join(tb_str), display=True)
            raise

    if not args.param_selection:
        setup_and_run_experiment(experiment_name, experiment_dir, log, train_ds, test_ds, reception, proto_len)
    else:
        log(f"Running hyperparameter selection for {uea_dataset if uea_dataset else 'custom dataset'}...", flush=True, display=True)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if uea_dataset == "StandWalkJump":
            kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        for reception in [0.25, 0.5, 0.75, 0.9]:
            for denom in [100, 10, 5, 2, 1]:
                log(f"Running for reception {reception} and proto_len {1/denom}", flush=True, display=True)
                for fold_idx, (train_ind, test_ind) in enumerate(kfold.split(train_ds.X, train_ds.y)):
                    proto_len = int(ts_len // denom)
                    if proto_len == 0:
                        continue

                    fold_train_ds = TSCDataset(train_ds.X[train_ind], train_ds.y[train_ind])
                    fold_test_ds = TSCDataset(train_ds.X[test_ind], train_ds.y[test_ind])

                    fold_experiment_name = f"{experiment_name}/reception-{reception}/proto_len-{1/denom}/fold-{fold_idx}"
                    fold_experiment_dir = experiment_setup(fold_experiment_name, args.target_protos_dir)
                    fold_log, fold_logclose = create_logger(fold_experiment_dir / "log.txt", display=args.verbose)

                    curr_link_path = experiment_dir / "curr_log.txt"
                    if os.path.islink(curr_link_path):
                        os.unlink(curr_link_path)
                    os.symlink(fold_experiment_dir / "log.txt", curr_link_path)

                    setup_and_run_experiment(fold_experiment_name, fold_experiment_dir, fold_log, fold_train_ds, fold_test_ds, reception, proto_len)

                    fold_logclose()
                    os.unlink(curr_link_path)
