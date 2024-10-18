from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import contextlib
import joblib
from typing import Union, List, Dict
from scipy import signal
from dataclasses import dataclass


class TSCDataset():
    X: np.ndarray
    y: np.ndarray

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class TrainTestDS:
    name: str
    train: TSCDataset
    test: TSCDataset
    val: TSCDataset = None


@dataclass
class DSInfo:
    name: str
    features: int
    ts_len: int
    num_classes: int


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class ArtificialProtos():
    def __init__(self, N, feature_noise_power=0.1, randomize_right_side=False):
        self.data = []
        x = np.linspace(0, 100, 100)
        for _ in range(N):
            label = np.random.randint(0, 4)
            ts = np.zeros((3, 100))
            if label == 0:
                ts[0, :40] = signal.sawtooth(x[:40] / (1+1))
                ts[1, :40] = signal.sawtooth(x[:40] / (2+1))
            elif label == 1:
                ts[0, :40] = signal.sawtooth(x[:40] / (1+1))
                ts[1, :40] = signal.square(x[:40] / (2+1))
            elif label == 2:
                ts[0, :40] = signal.square(x[:40] / (1+1))
                ts[1, :40] = signal.sawtooth(x[:40] / (2+1))
            else:
                ts[0, :40] = signal.square(x[:40] / (1+1))
                ts[1, :40] = signal.square(x[:40] / (2+1))
            if np.random.choice([0, 1]) == 0:
                ts[2, :40] = signal.square(np.random.choice([-1, 1]) * x[:40] / 3)
            else:
                ts[2, :40] = signal.sawtooth(np.random.choice([-1, 1]) * x[:40] / 3)
            for i in range(3):
                if randomize_right_side:
                    ts[i, 40:] = np.sin(x[40:] / (np.random.randint(0, 4)+i+1)) / 3
                else:
                    ts[i, 40:] = np.sin(x[40:] / (i+1)) / 3
                ts[i, :] += np.random.normal(0, feature_noise_power, 100)
            self.data.append((ts.astype('float32'), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ArtificialProtosMoreFeatures():
    def __init__(self, N, feature_noise_power=0.1, meaningful_features=4, meaningless_features=3):
        self.data = []
        self.meaningful_features = meaningful_features
        self.meaningless_features = meaningless_features
        x = np.linspace(0, 100, 100)
        for _ in range(N):
            label = np.random.randint(0, 4)
            ts = np.zeros((meaningful_features+meaningless_features, 100))
            if label == 0:
                for i in range(meaningful_features):
                    ts[i, :40] = signal.square(x[:40] / (i+1))
            elif label == 1:
                for i in range(meaningful_features):
                    ts[i, :40] = signal.sawtooth(x[:40] / (i+1))
            elif label == 2:
                for i in range(meaningful_features):
                    ts[i, :40] = 1 / (1 + np.exp(-x[:40] / (i+1)))  # sigmoid function
            else:
                for i in range(meaningful_features):
                    ts[i, :40] = np.arctan(x[:40] / (i+1))
            for i in range(meaningful_features):
                ts[i, 40:] = np.sin(x[40:] / (i+1))
                ts[i, :] += np.random.normal(0, feature_noise_power, 100)
            for i in range(meaningful_features, meaningful_features+meaningless_features):
                r = np.random.randint(0, 3)
                ts[i, :] = signal.square(x / (i+1)) if r == 0 else signal.sawtooth(x / (i+1)) if r == 1 else np.sin(x / (i+1))
            self.data.append((ts.astype('float32'), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def transform_ts_data(X, scaler, scale_separately=True):
    if scale_separately:
        for i in range(X.shape[1]):
            X[:, i, :] = scaler.fit_transform(X[:, i, :])
        return X
    else:
        original_shape = X.shape
        X = X.reshape(-1, original_shape[2])
        X = scaler.fit_transform(X)
        X = X.reshape(original_shape)
        return X


def load_arff_dataset(datasets_path, dataset_name, train_size=None, val_size=None, scaler=None, scale_separately=True):
    train_file = datasets_path / dataset_name / f'{dataset_name}_TRAIN.arff'
    test_file = datasets_path / dataset_name / f'{dataset_name}_TEST.arff'
    label_encoder = LabelEncoder()

    def arff_to_numpy(file_path):
        data, meta = loadarff(file_path)
        X, y = [], []
        for row in data:
            x, label = row
            X.append(np.array(x.tolist(), dtype='float32'))
            y.append(label)
        return np.nan_to_num(np.stack(X, axis=0), 0), np.stack(y, axis=0)
    
    if train_size is None:
        trainX, trainy = arff_to_numpy(train_file)
        testX, testy = arff_to_numpy(test_file)
        trainy = label_encoder.fit_transform(trainy)
        testy = label_encoder.transform(testy)
    else:
        X1, y1 = arff_to_numpy(train_file)
        X2, y2 = arff_to_numpy(test_file)
        X = np.concatenate([X1, X2])
        y = np.concatenate([y1, y2])
        y = label_encoder.fit_transform(y)
        trainX, testX, trainy, testy = train_test_split(X, y, train_size=train_size)

    if val_size:
        trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=val_size)

    if scaler:
        trainX = transform_ts_data(trainX, scaler, scale_separately)
        testX = transform_ts_data(testX, scaler, scale_separately)

    return TrainTestDS(dataset_name, TSCDataset(trainX, trainy), TSCDataset(testX, testy), TSCDataset(valX, valy) if val_size else None)


def ds_load(datasets_path, ds_or_lst, train_size=None, val_size=None, scaler=None, scale_separately=True, get_info=False) -> Union[TrainTestDS, Dict[str, TrainTestDS]]:
    # scaler is assumed to reset its state when calling fit_transform/fit
    def load_ds(ds):
        return (ds, load_arff_dataset(datasets_path, ds, train_size, val_size, scaler, scale_separately))

    ds_names = ds_or_lst
    if not isinstance(ds_names, (list, tuple)):
        ds_names = [ds_names]

    with _tqdm_joblib(tqdm(desc="Loading datasets", total=len(ds_names))) as progress_bar:
        results = Parallel(n_jobs=32)(delayed(load_ds)(ds) for ds in ds_names)

    results = dict(results)

    if get_info:
        info = ds_get_info(ds_or_lst)
        if not isinstance(ds_or_lst, (list, tuple)):
            return results[ds_or_lst], info
        else:
            return {ds_name: (results[ds_name], info[ds_name]) for ds_name in results}

    if not isinstance(ds_or_lst, (list, tuple)):
        return results[ds_or_lst]
    else:
        return results


def ds_get_info(ds_or_lst=None) -> Union[List[DSInfo], DSInfo]:
    ds_names = ds_or_lst
    if not isinstance(ds_names, (list, tuple)) and ds_names:
        ds_names = [ds_names]

    datasets_info = pd.read_csv('DataDimensionsPipe.csv', delimiter='|', index_col=0)
    datasets_info.loc['PhonemeSpectra'] = datasets_info.loc['Phoneme']
    datasets_info = datasets_info.drop('Phoneme').sort_index()
    if ds_names:
        datasets_info = datasets_info.loc[ds_names]

    ret = {}
    for i, row in datasets_info.iterrows():
        ret[i] = DSInfo(i, row['NumDimensions'], row['SeriesLength'], row['NumClasses'])

    if not isinstance(ds_or_lst, (list, tuple)) and ds_or_lst:
        return ret[ds_or_lst]
    else:
        return ret
