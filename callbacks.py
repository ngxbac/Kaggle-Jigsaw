import torch.nn.functional as F
import numpy as np
from catalyst.dl.callbacks.base import *
from utils import *


class JigsawLossCallback(Callback):
    def __init__(
        self,
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state, criterion):
        output_bin = state.output['output_bin']
        output_aux = state.output['output_aux']

        target_bin = state.input['y']
        target_aux = state.input['y_aux']
        weight_bin = state.input['X_weight']

        # import pdb
        # pdb.set_trace()

        loss = criterion(
            output_bin=output_bin,
            target_bin=target_bin,
            weight_bin=weight_bin,
            output_aux=output_aux,
            target_aux=target_aux
        )
        return loss

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class AucCallback(Callback):
    """
    Auc metric callback.
    """

    def __init__(self,
                 identity,
                 target,
                 output_key: str = "output_bin"):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.output_key = output_key
        self.identity_valid = identity
        self.target = target

    def on_loader_start(self, state):
        self.outputs = []

    def on_batch_end(self, state):
        output = F.sigmoid(state.output[self.output_key])
        output = output.detach().cpu().numpy()

        self.outputs.append(output)

    def on_loader_end(self, state):
        if state.loader_name == 'valid':
            import warnings
            warnings.filterwarnings("ignore")

            self.outputs = np.concatenate(self.outputs, axis=0)

            total_score = scoring_valid(
                self.outputs,
                self.identity_valid,
                self.target,
                model_name="quora_multitarget",
                save_output=True
            )

            state.metrics.epoch_values[state.loader_name]['auc'] = total_score


class OptimizerCallbackJigsaw(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """

    def __init__(
        self,
        grad_clip_params: Dict = None,
        accumulation_steps: int = 1,
        optimizer_key: str = None,
        loss_key: str = None,
        prefix: str = None
    ):
        """
        @TODO: docs
        """

        grad_clip_params = grad_clip_params or {}
        self.grad_clip_fn = GRAD_CLIPPERS.get_from_params(**grad_clip_params)

        self.accumulation_steps = accumulation_steps
        self.optimizer_key = optimizer_key
        self.loss_key = loss_key
        self.prefix = prefix
        self._optimizer_wd = 0
        self._accumulation_counter = 0

    @staticmethod
    def grad_step(*, optimizer, optimizer_wd=0, grad_clip_fn=None):
        for group in optimizer.param_groups:
            if optimizer_wd > 0:
                for param in group["params"]:
                    param.data = param.data.add(
                        -optimizer_wd * group["lr"], param.data
                    )
            if grad_clip_fn is not None:
                grad_clip_fn(group["params"])
        optimizer.step()

    def on_stage_start(self, state: RunnerState):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        lr = optimizer.defaults["lr"]
        momentum = get_optimizer_momentum(optimizer)
        state.set_key(lr, "lr", inner_key=self.optimizer_key)
        state.set_key(momentum, "momentum", inner_key=self.optimizer_key)

    def on_epoch_start(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        self._optimizer_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
        optimizer.param_groups[0]["weight_decay"] = 0.0

    def on_batch_start(self, state):
        state.loss = None

    def on_batch_end(self, state):
        loss = state.get_key(key="loss", inner_key=self.loss_key)
        if isinstance(loss, dict):
            loss = list(loss.values())
        if isinstance(loss, list):
            loss = torch.mean(torch.stack(loss))

        if self.prefix is not None:
            state.metrics.add_batch_value(metrics_dict={
                self.prefix: loss.item(),
            })

        if not state.need_backward:
            return

        self._accumulation_counter += 1
        model = state.model
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )

        # This is very hacky check whether we have AMP optimizer and this may
        # change in future.
        # But alternative solution is to have AmpOptimizerCallback.
        # or expose another c'tor argument.
        if hasattr(optimizer, "_amp_stash"):
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self._accumulation_counter % self.accumulation_steps == 0:
            self.grad_step(
                optimizer=optimizer,
                optimizer_wd=self._optimizer_wd,
                grad_clip_fn=self.grad_clip_fn
            )
            model.zero_grad()
            self._accumulation_counter = 0

    def on_epoch_end(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        optimizer.param_groups[0]["weight_decay"] = self._optimizer_wd