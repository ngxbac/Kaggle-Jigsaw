from typing import Iterable, Any, Mapping, Dict, List, Tuple
from catalyst.dl.experiments import SupervisedRunner, BaseExperiment
from catalyst.dl.callbacks import Callback, LossCallback, OptimizerCallback, \
    SchedulerCallback, CheckpointCallback  # noqa F401
from callbacks import JigsawLossCallback, OptimizerCallbackJigsaw
from catalyst.dl.utils.utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from config import *

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


class JigsawExperiment(BaseExperiment):
    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks = self._callbacks
        if not stage.startswith("infer"):
            if config.use_bug:
                print("Using bug as features !!!")
                default_callbacks = [
                    (self._criterion, JigsawLossCallback),
                    (self._optimizer, OptimizerCallback),
                    (self._scheduler, SchedulerCallback),
                    ("_default_saver", CheckpointCallback),
                ]
            else:
                default_callbacks = [
                    (self._criterion, JigsawLossCallback),
                    (self._optimizer, OptimizerCallbackJigsaw),
                    (self._scheduler, SchedulerCallback),
                    ("_default_saver", CheckpointCallback),
                ]

            for key, value in default_callbacks:
                is_already_present = any(
                    isinstance(x, value) for x in callbacks)
                if key is not None and not is_already_present:
                    if key == self._optimizer:
                        callbacks.append(value(accumulation_steps=config.accumulation_steps))
                    else:
                        callbacks.append(value())
        return callbacks


def process_components(
    model: _Model,
    criterion: _Criterion = None,
    optimizer: _Optimizer = None,
    scheduler: _Scheduler = None,
    distributed_params: Dict = None
) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:

    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    device = UtilsFactory.get_device()

    if torch.cuda.is_available():
        cudnn.benchmark = True

    model = model.to(device)

    if is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        UtilsFactory.assert_fp16_available()
        from apex import amp

        distributed_rank = distributed_params.pop("rank", -1)

        # if distributed_rank > -1:
        #     torch.cuda.set_device(distributed_rank)
        #     torch.distributed.init_process_group(
        #         backend="nccl", init_method="env://")

        model, optimizer = amp.initialize(
            model, optimizer, **distributed_params)

        # if distributed_rank > -1:
            # from apex.parallel import DistributedDataParallel
            # model = DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    # criterion = criterion.to(device)
    # optimizer = optimizer.to(device)
    # scheduler = scheduler.to(device)

    return model, criterion, optimizer, scheduler, device


def trim_tensors(tsrs):
    max_len = torch.max(torch.sum((tsrs != 0), 1))
    if max_len > 2:
        tsrs = tsrs[:, :max_len]
    return tsrs


class ModelRunner(SupervisedRunner):
    _default_experiment = JigsawExperiment

    def _get_experiment_components(
        self,
        stage: str = None
    ) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """

        model = self.experiment.get_model(stage)
        criterion, optimizer, scheduler = \
            self.experiment.get_experiment_components(model, stage)

        model, criterion, optimizer, scheduler, device = \
            process_components(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                distributed_params=self.experiment.distributed_params
            )

        return model, criterion, optimizer, scheduler, device

    def _run_loader(self, loader):
        if loader.batch_sampler:
            batch_size = loader.batch_sampler.batch_size
        elif loader.sampler:
            batch_size = loader.sampler.batch_size
        else:
            batch_size = loader.batch_size
        self.state.batch_size = batch_size
        self.state.step = (
            self.state.step
            or self.state.epoch * len(loader) * self.state.batch_size
        )
        # @TODO: remove time usage, use it under the hood
        self.state.timer.reset()

        self.state.timer.start("_timers/batch_time")
        self.state.timer.start("_timers/data_time")

        for i, batch in enumerate(loader):
            batch = self._batch2device(batch, self.device)
            self.state.timer.stop("_timers/data_time")

            self._run_event("batch_start")

            self.state.timer.start("_timers/model_time")
            self._run_batch(batch)
            self.state.timer.stop("_timers/model_time")

            self.state.timer.stop("_timers/batch_time")
            self._run_event("batch_end")

            self.state.timer.reset()

            if self._check_run and i >= 3:
                break

            self.state.timer.start("_timers/batch_time")
            self.state.timer.start("_timers/data_time")

    def predict_batch(self, batch: Mapping[str, Any]):
        # batch['X'] = trim_tensors(batch['X'])
        output = self.model(
            input_ids=batch['X'],
            features=batch['X_meta'],
            attention_mask=(batch['X'] > 0),
            labels=None
        )

        return {
            "output_bin": output[:, :1],
            "output_aux": output[:, 1:]
        }
