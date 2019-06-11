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
            default_callbacks = [
                (self._criterion, JigsawLossCallback),
                (self._optimizer, OptimizerCallback),
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

    return model, criterion, optimizer, scheduler, device


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

    def predict_batch(self, batch: Mapping[str, Any]):
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
