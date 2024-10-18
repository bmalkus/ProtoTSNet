import numpy as np
from scipy import signal


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
