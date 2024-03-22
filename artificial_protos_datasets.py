import numpy as np
import torch
from scipy import signal


class ArtificialProtos:
    def __init__(self, N, feature_noise_power=0.1, randomize_right_side=False):
        self.X = []
        self.y = []
        x = np.linspace(0, 100, 100)
        for _ in range(N):
            label = np.random.randint(0, 4)
            ts = np.zeros((3, 100))
            if label == 0:
                ts[0, :40] = signal.sawtooth(x[:40] / (1 + 1))
                ts[1, :40] = signal.sawtooth(x[:40] / (2 + 1))
            elif label == 1:
                ts[0, :40] = signal.sawtooth(x[:40] / (1 + 1))
                ts[1, :40] = signal.square(x[:40] / (2 + 1))
            elif label == 2:
                ts[0, :40] = signal.square(x[:40] / (1 + 1))
                ts[1, :40] = signal.sawtooth(x[:40] / (2 + 1))
            else:
                ts[0, :40] = signal.square(x[:40] / (1 + 1))
                ts[1, :40] = signal.square(x[:40] / (2 + 1))
            if np.random.choice([0, 1]) == 0:
                ts[2, :40] = signal.square(np.random.choice([-1, 1]) * x[:40] / 3)
            else:
                ts[2, :40] = signal.sawtooth(np.random.choice([-1, 1]) * x[:40] / 3)
            for i in range(3):
                if randomize_right_side:
                    ts[i, 40:] = np.sin(x[40:] / (np.random.randint(0, 4) + i + 1)) / 3
                else:
                    ts[i, 40:] = np.sin(x[40:] / (i + 1)) / 3
                ts[i, :] += np.random.normal(0, feature_noise_power, 100)
            self.X.append(ts.astype("float32"))
            self.y.append(label)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
