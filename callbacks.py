import torch.nn.functional as F
import numpy as np
from catalyst.dl.callbacks.core import Callback
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