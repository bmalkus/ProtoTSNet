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
from autoencoder import (PermutingConvAutoencoder, RegularConvAutoencoder,
                         train_autoencoder)
from datasets_utils import TSCDataset, ds_load
from model import ProtoTSNet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torcheval.metrics.functional import multiclass_confusion_matrix
from train import EpochType, ProtoTSCoeffs, create_logger, train_prototsnet

device = torch.device('cuda')
# torch.cuda.set_per_process_memory_fraction(fraction=0.5, device=0)  # or 1, watch out for CUDA_VISIBLE_DEVICES

# dataset in arff format should be put in the 'datasets/' directory (downloaded from timeseriesclassification.com)
DATASETS_PATH = Path('datasets')

def experiment_setup(experiment_subpath):
    experiment_dir = Path.cwd() / 'experiments' / experiment_subpath
    os.makedirs(experiment_dir, exist_ok=True)

    shutil.copy(src=Path.cwd()/'autoencoder.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'datasets_utils.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'main.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'model.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'push.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'train_utils.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'train.py', dst=experiment_dir)

    return experiment_dir

parser = argparse.ArgumentParser(description='Run experiment with specified dataset and experiment directory.')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
parser.add_argument('--experiment_name', type=str, required=True, help='Directory to save experiment results')
parser.add_argument('--permuting_encoder', action='store_true', help='Use permuting encoder', required=False, default=True)
parser.add_argument('--encoder_pretraining', action='store_true', help='Train encoder before ProtoTSNet', required=False, default=True)
parser.add_argument('--pretraining_epochs', type=int, help='Number of encoder pretraining epochs', required=False, default=50)
parser.add_argument('--num_warm_epochs', type=int, help='Number of warm-up epochs', required=False, default=None)
parser.add_argument('--push_start_epoch', type=int, help='Epoch to start pushing prototypes', required=False, default=110)
parser.add_argument('--push_epochs_interval', type=int, help='Interval between pushing prototypes', required=False, default=30)
parser.add_argument('--last_layer_epochs', type=int, help='Number of epochs to train last layer', required=False, default=40)
parser.add_argument('--epochs', type=int, help='Number of epochs to train', required=False, default=200)
parser.add_argument('--proto_features', type=int, help='Number of latent features', required=False, default=32)
parser.add_argument('--proto_len', type=int, help='Prototype length (in time steps)', required=False, default=None)
parser.add_argument('--reception', type=float, help='Fraction of significant features', required=False, default=None)
parser.add_argument('--l1_addon_coeff', type=float, help='L1 regularization coefficient for feature importance layer', required=False, default=1e-3)
parser.add_argument('--l1_coeff', type=float, help='L1 regularization coefficient', required=False, default=1e-3)
parser.add_argument('--clst_coeff', type=float, help='Cluster separation coefficient', required=False, default=0.08)
parser.add_argument('--sep_coeff', type=float, help='Separation coefficient', required=False, default=-0.008)
parser.add_argument('--param_selection', action='store_true', help='Run hyperparameter selection', required=False, default=False)
parser.add_argument('--verbose', action='store_true', help='Verbose output', required=False, default=False)
args = parser.parse_args()

ds_name = args.dataset

experiment_name = f"{args.experiment_name}/{ds_name}"
experiment_dir = experiment_setup(experiment_name)
log, logclose = create_logger(experiment_dir / "log.txt", display=args.verbose)

log(f'Loading dataset {ds_name}...', flush=True, display=True)
train_ds, test_ds = ds_load(DATASETS_PATH, ds_name, scaler=StandardScaler())

# read best_params.csv
best_params = pd.read_csv('best_params.csv', index_col=0)

# hyperparameters
protos_per_class = 10  # number of prototypes will equal 'protos_per_class * number of classes'
proto_len = int(best_params.loc[ds_name, 'proto_len']) if args.proto_len is None else args.proto_len  # prototype length (number of time steps) - it is latent space length, so due to receptive field in the input space it is longer
proto_features = args.proto_features  # number of latent features (dimensions) that input is encoded to
reception = float(best_params.loc[ds_name, 'reception']) if args.reception is None else args.reception  # estimate for the fraction of significant features, better to underestimate than overestimate
permuting_encoder = args.permuting_encoder
encoder_pretraining = args.encoder_pretraining
num_warm_epochs = args.num_warm_epochs if args.num_warm_epochs is not None else 50 if encoder_pretraining else 0 # number of epochs during which encoder weights are frozen, value >0 only makes sense if encoder is pretrained
push_start_epoch = args.push_start_epoch  # when to start pushing prototypes onto the input data
push_epochs = range(push_start_epoch, 1000, args.push_epochs_interval)  # which epochs to push prototypes on
num_last_layer_epochs = args.last_layer_epochs  # how many epochs to train the last layer (prototypes <-> class mapping)
epochs = args.epochs  # overall number of epochs (PUSH + last layer "epochs" count as one epoch here), set it so that the training ends with PUSH

coeffs = ProtoTSCoeffs(crs_ent=1, l1_addon=args.l1_addon_coeff, l1=args.l1_coeff, clst=args.clst_coeff, sep=args.sep_coeff)  # how much each element contributes to the loss, l1 is last layer l1 regularization, l1_addon is regularization of feature importance layer

# retrieve details of the dataset
num_classes = len(np.unique(train_ds.y))
num_features = train_ds.X.shape[1]
ts_len = train_ds.X.shape[2]

def setup_and_run_experiment(experiment_name, experiment_dir, log, train_ds, test_ds, reception, proto_len):
    try:
        train_batch_size = 32
        # reduce in case dataset is small
        while train_batch_size > len(train_ds.X) / 4:
            train_batch_size //= 2
        test_batch_size = 128

        do_batch_norm = True
        if train_batch_size < 16:
            do_batch_norm = False

        whole_training_start = time.time()

        if permuting_encoder:
            autoencoder = PermutingConvAutoencoder(num_features=num_features, latent_features=proto_features, reception_percent=reception, padding='same', do_max_pool=False, do_batch_norm=do_batch_norm)
        else:
            autoencoder = RegularConvAutoencoder(num_features=num_features, latent_features=proto_features, padding='same', do_max_pool=False, do_batch_norm=do_batch_norm)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
        if encoder_pretraining:
            log(f'Training encoder', flush=True, display=True)
            train_autoencoder(autoencoder, train_loader, test_loader, device=device, log=log, num_epochs=args.pretraining_epochs)
        autoencoder.encoder.set_return_indices(False)

        ptsnet = ProtoTSNet(
            cnn_base=autoencoder.encoder,
            num_features=num_features,
            ts_sample_len=ts_len,
            proto_num=protos_per_class * num_classes,
            latent_features=proto_features,
            proto_len_latent=proto_len,
            num_classes=num_classes,
            prototype_activation_function='log',
        )

        def lr_sched_setup(optimizer, epoch_type):
            if epoch_type == EpochType.JOINT:
                return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=3e-2, step_size_up=10, step_size_down=20, mode='exp_range', gamma=0.99, cycle_momentum=False)
            elif epoch_type == EpochType.LAST_LAYER:
                return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=15, step_size_down=25, mode='exp_range', gamma=0.99, cycle_momentum=False)
            return None

        log(f'Training ProtoTSNet', flush=True, display=True)
        trainer = train_prototsnet(
            ptsnet=ptsnet,
            experiment_dir=experiment_dir,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            num_warm_epochs=num_warm_epochs,
            push_start_epoch=push_start_epoch,
            push_epochs=push_epochs,
            num_last_layer_epochs=num_last_layer_epochs,
            num_epochs=epochs,
            coeffs=coeffs,
            lr_sched_setup=lr_sched_setup,
            log=log,
            add_params_to_log={
                'encoder_pretraining': encoder_pretraining,
                'permuting_encoder': permuting_encoder,
                'reception': reception,
                'proto_len_fraction': f'{proto_len / ts_len:.3f}',
            }
        )

        accu_test = trainer.latest_stat("accu_test")
        log(f'Last epoch test accu: {accu_test*100:.2f}%', display=True)
        with open(experiment_dir / "test_accu.json", "w") as f:
            json.dump({"value": accu_test}, f, indent=4)

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
        confusion_matrix = torch.zeros(ptsnet.num_classes, ptsnet.num_classes)
        for i, (image, label) in enumerate(test_loader):
            output, _ = ptsnet(image.to(device))
            confusion_matrix += multiclass_confusion_matrix(output.to('cpu'), label, num_classes=output.shape[1])
        np.savetxt(experiment_dir / 'confusion_matrix.txt', confusion_matrix.numpy(), fmt='%4d')

        whole_training_end = time.time()
        log(f"Done in {trainer.curr_epoch - 1} epochs, {whole_training_end - whole_training_start:.2f}s", display=True)
    except Exception as e:
        log(f"Exception ocurred for experiment {experiment_name}: {e}", display=True)
        tb_str = traceback.format_tb(e.__traceback__)
        log('\n'.join(tb_str), display=True)
        raise

if not args.param_selection:
    setup_and_run_experiment(experiment_name, experiment_dir, log, train_ds, test_ds, reception, proto_len)
else:
    log(f'Running hyperparameter selection for {ds_name}...', flush=True, display=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if ds_name == 'StandWalkJump':
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    for reception in [0.25, 0.5, 0.75, 0.9]:
        for denom in [100, 10, 5, 2, 1]:
            log(f'Running for reception {reception} and proto_len {1/denom}', flush=True, display=True)
            for fold_idx, (train_ind, test_ind) in enumerate(kfold.split(train_ds.X, train_ds.y)):
                proto_len = int(ts_len // denom)
                if proto_len == 0:
                    continue

                fold_train_ds = TSCDataset(train_ds.X[train_ind], train_ds.y[train_ind])
                fold_test_ds = TSCDataset(train_ds.X[test_ind], train_ds.y[test_ind])

                fold_experiment_name = f"{experiment_name}/reception-{reception}/proto_len-{1/denom}/fold-{fold_idx}"
                fold_experiment_dir = experiment_setup(fold_experiment_name)
                fold_log, fold_logclose = create_logger(fold_experiment_dir / "log.txt", display=args.verbose)

                curr_link_path = experiment_dir / 'curr_log.txt'
                if os.path.islink(curr_link_path):
                    os.unlink(curr_link_path)
                os.symlink(fold_experiment_dir / "log.txt", curr_link_path)

                setup_and_run_experiment(fold_experiment_name, fold_experiment_dir, fold_log, fold_train_ds, fold_test_ds, reception, proto_len)

                fold_logclose()
                os.unlink(curr_link_path)

