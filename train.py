import json
import os
import shutil
import time
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from pathlib import Path

import push
import torch
from log import create_logger
from model import ProtoTSNet
from torch.utils.data import DataLoader
from train_utils import EarlyStopping, EpochType

ProtoTSCoeffs = namedtuple('ProtoTSCoeffs', 'crs_ent,clst,sep,l1,l1_addon', defaults=(1,0,0,0,0))


def train_prototsnet(
    ptsnet: ProtoTSNet,
    experiment_dir,
    device,
    coeffs,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_warm_epochs,
    push_start_epoch,
    push_epochs,
    num_last_layer_epochs,
    num_epochs=1000,
    class_specific=True,
    val_loader: DataLoader = None,
    learning_rates=None,
    features_lr=1e-3,  # effective only if learning_rates is None
    lr_sched_setup=None,
    custom_hooks=None,
    early_stopper=None,
    log=None
):
    save_files = True if experiment_dir is not None else False
    logclose = lambda: None
    if save_files:
        if not isinstance(experiment_dir, Path):
            experiment_dir = Path(experiment_dir)
        models_dir = experiment_dir / 'models'
        os.makedirs(models_dir, exist_ok=True)
        proto_dir = experiment_dir / 'protos'
        os.makedirs(proto_dir, exist_ok=True)
        if log is None:
            log, logclose = create_logger(experiment_dir / "log.txt", display=True)
    else:
        models_dir = None
        proto_dir = None
        log = print

    try:
        learning_rates = {
            EpochType.JOINT: {
                'features': features_lr,
                'add_on_layers': 3e-3,
                'prototype_vectors': 3e-3
            },
            EpochType.WARM: {
                'add_on_layers': 3e-3,
                'prototype_vectors': 3e-3
            },
            EpochType.LAST_LAYER: {
                'add_on_layers': 1e-3
            }
        }

        params = {
            "proto_num": ptsnet.prototype_shape[0],
            "proto_features": ptsnet.prototype_shape[1],
            "proto_len_latent": ptsnet.prototype_shape[2],
            "protos_per_class": ptsnet.prototype_shape[0] // ptsnet.num_classes if class_specific else None,
            "ts_len": ptsnet.ts_sample_len,
            "num_classes": ptsnet.num_classes,
            "coeffs": coeffs._asdict(),
            "num_warm_epochs": num_warm_epochs,
            "push_start_epoch": push_start_epoch,
            "num_last_layer_epochs": num_last_layer_epochs,
            "epochs": num_epochs,
            "learning_rates": str(learning_rates),
        }
        with open(experiment_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        trainer = ProtoTSNetTrainer(
            ptsnet,
            device,
            train_loader,
            test_loader,
            val_loader,
            num_epochs=num_epochs,
            num_warm_epochs=num_warm_epochs,
            push_start_epoch=push_start_epoch,
            push_epochs=push_epochs,
            num_last_layer_epochs=num_last_layer_epochs,
            coeffs=coeffs,
            learning_rates=learning_rates,
            lr_sched_setup=lr_sched_setup,
            class_specific=class_specific,
            proto_save_dir=proto_dir,
            early_stopper=early_stopper,
            hooks=[
                lambda t,_: t.dump_stats(experiment_dir / 'stats.json'),
                get_verbose_logger()
            ] + ([] if custom_hooks is None else custom_hooks),
            log=log,
        )

        if save_files:
            shutil.rmtree(proto_dir)
            os.makedirs(proto_dir, exist_ok=True)

        trainer.train()

        if save_files:
            torch.save(ptsnet.state_dict(), models_dir / f'last-epoch-state-dict.pth')
            torch.save(ptsnet, models_dir / f'last-epoch.pth')
            trainer.dump_stats(experiment_dir / 'stats.json')

        return trainer
    finally:
        logclose()


def best_stat_saver(stat_name, file_path, mode='min'):
    best = None
    push_best = None
    can_save_overal = False
    cmp = lambda a,b: a < b if mode == 'min' else a > b
    def save_best_stat(t, _):
        nonlocal best, push_best, can_save_overal

        found_better = False
        curr_loss = t.latest_stat(stat_name, raw=True)
        if t.curr_epoch_type in [EpochType.PUSH, EpochType.LAST_LAYER]:
            can_save_overal = True
            if push_best is None or cmp(curr_loss['value'], push_best['value']):
                push_best = curr_loss
                found_better = True

        if best is None or cmp(curr_loss['value'], best['value']):
            best = curr_loss
            found_better = True

        if found_better:
            with open(file_path, 'w') as f:
                json.dump({
                    'push': push_best,
                    'overall': best if can_save_overal else None
                }, f)
    return save_best_stat


def get_verbose_logger():
    def verbose_log_epoch(t: ProtoTSNetTrainer, _):
        last_layer_str = f" ({t.curr_last_layer_epoch}/{t.num_last_layer_epochs})" if t.curr_epoch_type == EpochType.LAST_LAYER else ""
        if t.val_loader is not None:
            t.log(f"epoch: {t.curr_epoch:3d}{last_layer_str} ({t.curr_epoch_type.name})")
            t.log(f"    {'val acc:':25s} {t.latest_stat('accu_val')*100:.2f}%")
            t.log(f"    {'train overall loss:':25s} {t.latest_stat('loss_train')}")
            t.log(f"    {'train cross_ent loss:':25s} {t.latest_stat('cross_ent_train')}")
            t.log(f"    {'val overall loss:':25s} {t.latest_stat('loss_val')}")
            t.log(f"    {'val cross_ent loss:':25s} {t.latest_stat('cross_ent_val')}")
            t.log(f"    {'cluster loss:':25s} {t.latest_stat('cluster_val')}")
            t.log(f"    {'separation loss:':25s} {t.latest_stat('separation_val')}")
            t.log(f"    {'avg separation loss:':25s} {t.latest_stat('avg_separation_val')}")
            t.log(f"    {'l1_addon loss:':25s} {t.latest_stat('l1_addon_val')}")
            t.log(f"    {'l1 loss:':25s} {t.latest_stat('l1_val')}")
            t.log(f"    {'train time:':25s} {t.latest_stat('time_train')}")
            t.log(f"    {'val time:':25s} {t.latest_stat('time_val')}")
            t.log(f"    {'epoch time:':25s} {t.latest_stat('epoch_time')}", flush=True)
            if t.curr_epoch_type == EpochType.JOINT:
                t.log(f"    {'joint lr:':25s} {t.latest_stat('joint_lr')}", flush=True)
            elif t.curr_epoch_type == EpochType.LAST_LAYER:
                t.log(f"    {'last layer lr:':25s} {t.latest_stat('last_layer_lr')}", flush=True)
            if t.latest_stat('did_run_test'):
                t.log(f"    Testing:")
                t.log(f"    {'test acc:':25s} {t.latest_stat('accu_test')*100:.2f}%")
                t.log(f"    {'test overall loss:':25s} {t.latest_stat('loss_test')}")
                t.log(f"    {'test cross_ent loss:':25s} {t.latest_stat('cross_ent_test')}")
                t.log(f"    {'test time:':25s} {t.latest_stat('time_test')}")
        else:
            t.log(f"epoch: {t.curr_epoch:3d}{last_layer_str} ({t.curr_epoch_type.name})")
            t.log(f"    {'test acc:':25s} {t.latest_stat('accu_test')*100:.2f}%")
            t.log(f"    {'train overall loss:':25s} {t.latest_stat('loss_train')}")
            t.log(f"    {'train cross_ent loss:':25s} {t.latest_stat('cross_ent_train')}")
            t.log(f"    {'train cluster loss:':25s} {t.latest_stat('cluster_train')}")
            t.log(f"    {'train separation loss:':25s} {t.latest_stat('separation_train')}")
            t.log(f"    {'train avg sep loss:':25s} {t.latest_stat('avg_separation_train')}")
            t.log(f"    {'train l1_addon loss:':25s} {t.latest_stat('l1_addon_train')}")
            t.log(f"    {'train l1 loss:':25s} {t.latest_stat('l1_train')}")
            t.log(f"    {'test overall loss:':25s} {t.latest_stat('loss_test')}")
            t.log(f"    {'test cross_ent loss:':25s} {t.latest_stat('cross_ent_test')}")
            t.log(f"    {'train time:':25s} {t.latest_stat('time_train')}")
            t.log(f"    {'test time:':25s} {t.latest_stat('time_test')}")
            t.log(f"    {'epoch time:':25s} {t.latest_stat('epoch_time')}", flush=True)
            if t.curr_epoch_type == EpochType.JOINT:
                t.log(f"    {'joint lr:':25s} {t.latest_stat('joint_lr')}", flush=True)
            elif t.curr_epoch_type == EpochType.LAST_LAYER:
                t.log(f"    {'last layer lr:':25s} {t.latest_stat('last_layer_lr')}", flush=True)
    return verbose_log_epoch


class ProtoTSNetTrainer:
    def __init__(
        self,
        ptsnet,
        device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs,
        num_warm_epochs,
        push_start_epoch,
        push_epochs,
        num_last_layer_epochs,
        coeffs: ProtoTSCoeffs,
        learning_rates,
        lr_sched_setup=None,
        class_specific=True,
        proto_save_dir=None,
        early_stopper=None,
        hooks=None,
        log=print,
    ):
        self.ptsnet = ptsnet
        self.device = device
        self.ptsnet.to(self.device)

        self.class_specific = class_specific

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.optimizers = {}
        self.lr_schedulers = {}

        self.proto_save_dir = proto_save_dir

        self.early_stopper = early_stopper

        self.hooks = hooks
        if self.hooks is None:
            self.hooks = []

        self.num_train_epochs = num_epochs
        self.num_warm_epochs = num_warm_epochs
        self.push_start = push_start_epoch
        if self.push_start == 'auto':
            self.push_epoch_monitor = EarlyStopping(patience=10, retrieve_stat='loss_val', mode='min', wait=11, eps=1e-2)
        else:
            self.push_epoch_monitor = None
        self.did_st_push = False
        self.logged_push_condition_met = False
        self.push_epochs = push_epochs
        self.num_last_layer_epochs = num_last_layer_epochs

        self.coeffs = coeffs._asdict()

        self._stats = defaultdict(list)
        self.curr_epoch_type = None
        self.curr_epoch = 1
        self.curr_last_layer_epoch = 1
        self.curr_true_epoch = 1
        self.run_type_str = None  # 'train', 'test', 'val'

        self.log = log

        self._setup_optimizers(learning_rates, lr_sched_setup)

    def _setup_optimizers(self, learning_rates, lr_sched_setup):
        warm_optimizer_specs = [
            {'params': self.ptsnet.add_on_layers.parameters(), 'lr': learning_rates[EpochType.WARM]['add_on_layers']},
            {'params': self.ptsnet.prototype_vectors, 'lr': learning_rates[EpochType.WARM]['prototype_vectors']},
        ]
        joint_optimizer_specs = [
            {'params': self.ptsnet.features.parameters(), 'lr': learning_rates[EpochType.JOINT]['features']},
            {'params': self.ptsnet.add_on_layers.parameters(), 'lr': learning_rates[EpochType.JOINT]['add_on_layers']},
            {'params': self.ptsnet.prototype_vectors, 'lr': learning_rates[EpochType.JOINT]['prototype_vectors']},
        ]
        last_layer_optimizer_specs = [
            {'params': self.ptsnet.last_layer.parameters(), 'lr': learning_rates[EpochType.LAST_LAYER]['add_on_layers']}
        ]

        self.optimizers[EpochType.WARM] = torch.optim.Adam(warm_optimizer_specs)
        self.optimizers[EpochType.JOINT] = torch.optim.Adam(joint_optimizer_specs)
        self.optimizers[EpochType.LAST_LAYER] = torch.optim.Adam(last_layer_optimizer_specs)

        self.lr_schedulers[EpochType.WARM] = lr_sched_setup(self.optimizers[EpochType.WARM], EpochType.WARM)
        self.lr_schedulers[EpochType.JOINT] = lr_sched_setup(self.optimizers[EpochType.JOINT], EpochType.JOINT)
        self.lr_schedulers[EpochType.LAST_LAYER] = lr_sched_setup(self.optimizers[EpochType.LAST_LAYER], EpochType.LAST_LAYER)

    def _add_stat(self, stat_name, value):
        if self.run_type_str is not None and not stat_name.endswith(self.run_type_str):
            stat_name = f'{stat_name}_{self.run_type_str}'

        stat = {
            'true_epoch': self.curr_true_epoch,
            'epoch': self.curr_epoch,
            'epoch_type': self.curr_epoch_type.name,
            'value': value,
        }
        if self.curr_epoch_type == EpochType.LAST_LAYER:
            stat['last_layer_epoch'] = self.curr_last_layer_epoch
        self._stats[stat_name].append(stat)

    def latest_stat(self, stat_name, raw=False):
        if stat_name not in self._stats:
            return None
        s = self._stats[stat_name][-1]
        return s if raw else s.get('value')

    def latest_stats(self):
        return {n: lst[-1] for n, lst in self._stats.items()}

    def _latest_stats_values(self):
        return {stat: vals[-1].get('value') for stat, vals in self._stats.items()}

    def stats(self):
        return dict(self._stats)

    def dump_stats(self, path):
        tmp_path = str(path) + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self._stats, f)
        os.rename(tmp_path, path)

    def _call_hooks(self):
        for hook in self.hooks:
            hook(self, self.ptsnet)

    @contextmanager
    def report_epoch_summary(self):
        start = time.time()
        yield
        end = time.time()
        self._add_stat('epoch_time', end - start)
        self._add_stat('joint_lr', self.lr_schedulers[EpochType.JOINT]._last_lr[0])
        self._add_stat('last_layer_lr', self.lr_schedulers[EpochType.LAST_LAYER]._last_lr[0])

    def _warm_epoch(self):
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.WARM)

            self._single_train_round()
            self._single_validation_round()
            self._single_test_round()

            if self.lr_schedulers[EpochType.WARM] is not None:
                self.lr_schedulers[EpochType.WARM].step()

    def _joint_epoch(self):
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.JOINT)

            self._single_train_round()
            self._single_validation_round()
            self._single_test_round()

            if self.lr_schedulers[EpochType.JOINT] is not None:
                self.lr_schedulers[EpochType.JOINT].step()

    def _push_protos(self):
        self.curr_true_epoch += 1
        with self.report_epoch_summary():
            # require_grad does not matter here, as we are not training, only pushing protos and testing
            self._set_epoch_type(EpochType.PUSH)

            prototype_ts_filename_prefix = 'prototype-ts'
            prototype_self_act_filename_prefix = 'prototype-self-act'
            proto_bounds_filename_prefix = 'bounds'

            push.push_prototypes(
                self.train_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network=self.ptsnet, # pytorch network with prototype_vectors
                class_specific=self.class_specific,
                preprocess_input_function=None,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=self.proto_save_dir, # if not None, prototypes will be saved here
                epoch_number=self.curr_epoch, # if not provided, prototypes saved previously will be overwritten
                proto_ts_filename_prefix=prototype_ts_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bounds_filename_prefix,
                save_prototype_class_identity=True,
                device=self.device)

            self._single_validation_round()
            self._single_test_round()

    def _optimize_last_layer(self):
        self._set_epoch_type(EpochType.LAST_LAYER)

        for self.curr_last_layer_epoch in range(1, self.num_last_layer_epochs+1):
            self.curr_true_epoch += 1
            with self.report_epoch_summary():
                self._single_train_round()
                self._single_validation_round()
                self._single_test_round()

                if self.lr_schedulers[EpochType.LAST_LAYER] is not None:
                    self.lr_schedulers[EpochType.LAST_LAYER].step()

            self._call_hooks()

    def _set_epoch_type(self, epoch_type: EpochType):
        self.curr_epoch_type = epoch_type
        self.ptsnet.features.set_requires_grad(epoch_type in [EpochType.JOINT])
        for p in self.ptsnet.add_on_layers.parameters():
            p.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM]
        self.ptsnet.prototype_vectors.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM]
        for p in self.ptsnet.last_layer.parameters():
            p.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM, epoch_type.LAST_LAYER]

    def _single_train_round(self):
        self.run_type_str = 'train'
        self.ptsnet.train()
        self._train_or_test(self.train_loader, optimizer=self.optimizers[self.curr_epoch_type])
        self.run_type_str = None

    def _single_validation_round(self):
        if self.val_loader is None:
            return
        self.run_type_str = 'val'
        self.ptsnet.eval()
        self._train_or_test(self.val_loader, optimizer=None)
        self.run_type_str = None

    def _single_test_round(self):
        self.run_type_str = 'test'
        if not self._test_round_cond():
            self._add_stat('did_run', False)
            self.run_type_str = None
            return
        self._add_stat('did_run', True)
        self.ptsnet.eval()
        self._train_or_test(self.test_loader, optimizer=None)
        self.run_type_str = None

    def _test_round_cond(self):
        if self.val_loader is not None:
            if self.curr_epoch % 10 != 0:
                return False
            if self.curr_epoch_type == EpochType.LAST_LAYER and self.curr_last_layer_epoch % 10 != 0:
                return False
        return True

    def _train_or_test(self, dataloader, optimizer=None, use_l1_mask=True):
        is_train = optimizer is not None
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0
        total_cluster_cost = 0
        # separation cost is meaningful only for class_specific
        total_separation_cost = 0
        total_avg_separation_cost = 0
        total_loss = 0

        for i, (image, label) in enumerate(dataloader):
            input = image.to(self.device)
            target = label.to(self.device)

            # torch.enable_grad() has no effect outside of no_grad()
            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                output, min_distances = self.ptsnet(input)

                # compute loss
                cross_entropy = torch.nn.functional.cross_entropy(output, target)

                if self.class_specific:
                    max_dist = (self.ptsnet.prototype_shape[1]
                                * self.ptsnet.prototype_shape[2])

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(self.ptsnet.prototype_class_identity[:,label]).to(self.device)
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=0)
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=0)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    # calculate avg cluster cost
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=0) / torch.sum(prototypes_of_wrong_class, dim=0)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    if use_l1_mask:
                        l1_mask = 1 - torch.t(self.ptsnet.prototype_class_identity).to(self.device)
                        l1 = (self.ptsnet.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = self.ptsnet.last_layer.weight.norm(p=1)

                    l1_addon = self.ptsnet.add_on_layers[0].weight.norm(p=1)
                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = self.ptsnet.last_layer.weight.norm(p=1)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                n_batches += 1
                total_cross_entropy += cross_entropy.item()
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

            # compute gradient and do SGD step
            if self.class_specific:
                if self.coeffs is not None:
                    loss = (self.coeffs['crs_ent'] * cross_entropy
                        + self.coeffs['clst'] * cluster_cost
                        + self.coeffs['sep'] * separation_cost
                        + self.coeffs['l1'] * l1
                        + self.coeffs['l1_addon'] * l1_addon)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if self.coeffs is not None:
                    loss = (self.coeffs['crs_ent'] * cross_entropy
                        + self.coeffs['clst'] * cluster_cost
                        + self.coeffs['l1'] * l1
                        + self.coeffs['l1_addon'] * l1_addon)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1

            total_loss += loss.item()
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            del input
            del target
            del output
            del predicted
            del min_distances

        end = time.time()

        self._add_stat('time', end - start)
        self._add_stat('loss', total_loss / n_batches)
        self._add_stat('cross_ent', total_cross_entropy / n_batches)
        self._add_stat('cluster', total_cluster_cost / n_batches)
        if self.class_specific:
            self._add_stat('separation', total_separation_cost / n_batches)
            self._add_stat('avg_separation', total_avg_separation_cost / n_batches)
        self._add_stat('accu', n_correct / n_examples)
        if use_l1_mask:
            l1_mask = 1 - torch.t(self.ptsnet.prototype_class_identity).to(self.device)
            l1 = (self.ptsnet.last_layer.weight * l1_mask).norm(p=1).item()
        else:
            l1 = self.ptsnet.last_layer.weight.norm(p=1).item()
        self._add_stat('l1', l1)
        self._add_stat('l1_addon', self.ptsnet.add_on_layers[0].weight.norm(p=1).item())

    def _st_push_condition(self):
        if self.push_epoch_monitor is not None:
            if self.push_epoch_monitor(self):
                if not self.logged_push_condition_met:
                    self.log(f'First push epoch condition met at epoch {self.curr_epoch}')
                    self.logged_push_condition_met = True
                    if self.early_stopper is not None:
                        self.early_stopper.stop_waiting()
                return True
            return False
        return self.curr_epoch >= self.push_start

    def _warm_epoch_condition(self):
        if self.num_warm_epochs == 'auto':
            return not self.did_st_push
        else:
            return self.curr_epoch <= self.num_warm_epochs

    def train(self):
        self.log('Starting training')
        t_start = time.time()

        while self.curr_epoch <= self.num_train_epochs:
            if self._warm_epoch_condition():
                self._warm_epoch()
            else:
                self._joint_epoch()
            self._call_hooks()

            if self._st_push_condition() and self.curr_epoch in self.push_epochs:
                self.did_st_push = True
                self._push_protos()
                self._call_hooks()

                if self.ptsnet.prototype_activation_function != 'linear':
                    self._optimize_last_layer()

            self.curr_epoch += 1
            self.curr_true_epoch += 1

            if self.early_stopper and self.early_stopper(self):
                self.log(f'Validation loss did not improve in {self.early_stopper.patience} epochs, aborting')
                break

        t_end = time.time()
        self.log(f'Finished training in {t_end - t_start:.2f} seconds')
        self._add_stat('total_time', t_end - t_start)
