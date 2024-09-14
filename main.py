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
from datasets_utils import ds_load
from log import create_logger
from model import ProtoTSNet
from sklearn.preprocessing import StandardScaler
from torcheval.metrics.functional import multiclass_confusion_matrix
from train import EpochType, ProtoTSCoeffs, train_prototsnet

device = torch.device('cuda')
torch.cuda.set_per_process_memory_fraction(fraction=0.5, device=0)  # or 1, watch out for CUDA_VISIBLE_DEVICES

# dataset in arff format should be put in the 'datasets/' directory (downloaded from timeseriesclassification.com)
DATASETS_PATH = Path('datasets')

def experiment_setup(experiment_subpath):
    experiment_dir = Path.cwd() / 'experiments' / experiment_subpath
    os.makedirs(experiment_dir, exist_ok=True)

    shutil.copy(src=Path.cwd()/'main.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'autoencoder.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'datasets_utils.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'experiments.ipynb', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'model.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'push.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'train_utils.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'train.py', dst=experiment_dir)
    
    return experiment_dir

ds_name = 'ArticularyWordRecognition'

experiment_name = f"{ds_name}TestExperimentScaling"
experiment_dir = experiment_setup(experiment_name)
log, logclose = create_logger(experiment_dir / "log.txt", display=True)

log(f'Loading dataset...', flush=True, display=True)
train_ds, test_ds = ds_load(DATASETS_PATH, ds_name, scaler=StandardScaler())

# read best_params.csv
best_params = pd.read_csv('best_params.csv', index_col=0)

# hyperparameters
protos_per_class = 10  # number of prototypes will equal 'protos_per_class * number of classes'
proto_len = int(best_params.loc[ds_name, 'proto_len'])  # prototype length (number of time steps) - it is latent space length, so due to receptive field in the input space it is longer
proto_features = 32  # number of latent features (dimensions) that input is encoded to
reception = 0.25  # estimate for the fraction of significant features, better to underestimate than overestimate
permuting_encoder = False
encoder_pretraining = True
num_warm_epochs = 50 if encoder_pretraining else 0  # number of epochs during which encoder weights are frozen, value >0 only makes sense if encoder is pretrained
push_start_epoch = 110  # when to start pushing prototypes onto the input data
push_epochs = range(push_start_epoch, 1000, 30)  # which epochs to push prototypes on
num_last_layer_epochs = 40  # how many epochs to train the last layer (prototypes <-> class mapping)
epochs = 200  # overall number of epochs (PUSH + last layer "epochs" count as one epoch here), set it so that the training ends with PUSH

coeffs = ProtoTSCoeffs(crs_ent=1, l1=1e-3, clst=0.08, sep=-0.008)  # how much each element contributes to the loss, l1 is last layer l1 regularization, l1_addon is regularization of feature importance layer

# retrieve details of the dataset
num_classes = len(np.unique(train_ds.y))
num_features = train_ds.X.shape[1]
ts_len = train_ds.X.shape[2]

train_batch_size = 32
# reduce in case dataset is small
while train_batch_size > len(train_ds.X) / 2:
    train_batch_size //= 2
test_batch_size = 128

try:
    whole_training_start = time.time()

    if permuting_encoder:
        autoencoder = PermutingConvAutoencoder(num_features=num_features, latent_features=proto_features, reception_percent=reception, padding='same')
        encoder = autoencoder.encoder
    else:
        autoencoder = RegularConvAutoencoder(num_features=num_features, latent_features=proto_features, padding='same', do_max_pool=False)
        encoder = autoencoder.encoder
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
    if encoder_pretraining:
        log(f'Training encoder', flush=True, display=True)
        train_autoencoder(autoencoder, train_loader, test_loader, device=device, log=log, num_epochs=50)
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
        log=log
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
finally:
    logclose()
