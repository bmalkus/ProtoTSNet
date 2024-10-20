from enum import Enum
from pathlib import Path

import torch

EpochType = Enum('EpochType', 'WARM,JOINT,PUSH,LAST_LAYER')


class BestModelCheckpointer:
    def __init__(self, stat_name, mode='min', add_save_cond=None, model_save_path=None, model_name=None):
        if model_name is None:
            self._model_name = 'saved_model'
        else:
            self._model_name = model_name
        self._model_save_path = Path(model_save_path) if model_save_path is not None else None
        self._is_better = (lambda a, b: a < b) if mode == 'min' else (lambda a, b: a > b)
        self._best_model = None
        self._stat_name = stat_name
        self._best = float('inf') if mode == 'min' else float('-inf')
        self._best_stats = None
        self._add_save_cond = add_save_cond

    def step(self, trainer, model):
        v = trainer.latest_stat(self._stat_name)
        if self._is_better(v, self._best):
            if callable(self._add_save_cond) and not self._add_save_cond(trainer, model):
                return
            self._best = v
            self._best_stats = trainer.latest_stats()
            self._save_best(model)

    def __call__(self, trainer, model=None):
        return self.step(trainer, model)

    def _save_best(self, model):
        if self._model_save_path:
            torch.save(model.state_dict(), self._model_save_path / f'{self._model_name}-state-dict.pth')
            torch.save(model, self._model_save_path / f'{self._model_name}.pth')
        else:
            self._best_model = model.state_dict()

    @property
    def best_model(self):
        if self._model_save_path:
            torch.load(self._model_save_path / f'{self._model_name}.pth')
        else:
            return self._best_model

    @property
    def best_value(self):
        return self._best

    @property
    def best_stats(self):
        return self._best_stats

    @property
    def should_stop(self):
        return self._should_stop


class EarlyStopping:
    def __init__(self, patience, retrieve_stat=None, mode='min', wait=0, model_save_path=None, model_name=None, eps=1e-5):
        self._patience = patience
        self._wait = wait
        self._stat_name = retrieve_stat
        if model_name is None:
            model_name = 'saved_model'
        self._model_save_path = (Path(model_save_path) / f'{model_name}.pth' if model_save_path is not None else None)
        self._eps = eps if eps is not None else 1e-5
        self._is_better = (lambda a, b: a < b - self._eps) if mode == 'min' else (lambda a, b: a > b + self._eps)
        self._counter = 0
        self._best_model = None

        self._should_stop = False
        self._best = float('inf') if mode == 'min' else float('-inf')

    def step(self, v_or_trainer, model=None):
        if self._should_stop:
            return True

        if self._wait == 'auto':
            return False

        if self._wait > 0:
            self._wait -= 1
            return False

        if self._stat_name:
            v = v_or_trainer.latest_stat(self._stat_name)
        else:
            v = v_or_trainer

        if self._is_better(v, self._best):
            self._best = v
            self._counter = 0
            if model:
                self._save_best(model)
        else:
            self._counter += 1
            if self._counter == self._patience:
                self._should_stop = True
                return True
        return False
    
    def stop_waiting(self):
        self._wait = 0

    def __call__(self, v_or_trainer, model=None):
        return self.step(v_or_trainer, model)

    def _save_best(self, model):
        if self._model_save_path:
            torch.save(model.state_dict(), self._model_save_path)
        else:
            self._best_model = model.state_dict()

    @property
    def patience(self):
        return self._patience

    @property
    def best_model(self):
        if self._model_save_path:
            torch.load(self._model_save_path)
        else:
            return self._best_model

    @property
    def best_value(self):
        return self._best

    @property
    def should_stop(self):
        return self._should_stop
