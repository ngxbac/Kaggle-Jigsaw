from pytorch_pretrained_bert.modeling_xlnet import XLNetPreTrainedModel, XLNetModel
from torch.nn import functional as F
import torch.nn as nn
import torch


class XLNetWithMultiOutput(XLNetPreTrainedModel):
    def __init__(self, config, clf_dropout, n_class):
        super(XLNetWithMultiOutput, self).__init__(config)
        self.transformer = XLNetModel(config)
        self.dropout = nn.Dropout(clf_dropout)

        dense_output = 2 * config.d_model + 64
        self.linear = nn.Linear(2 * config.d_model, n_class)

        added_hidden_size = 64
        # self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.added_linear = nn.Linear(11, added_hidden_size)
        # self.added_dropout = nn.Dropout(0.1)

        self.out = nn.Linear(dense_output, 1)
        self.aux_out = nn.Linear(dense_output, n_class)

        self.apply(self.init_xlnet_weights)

    def forward(self, input_ids, features, position_ids=None, token_type_ids=None, past=None, **kwargs):
        output, hidden_states, new_mems = self.transformer(input_ids, position_ids, token_type_ids, past)
        avg_pool = torch.mean(output, 1)
        max_pool, _ = torch.max(output, 1)

        added_fts = F.relu(self.added_linear(features))
        output1 = torch.cat((avg_pool, max_pool, added_fts), 1)

        output1 = self.dropout(output1)

        out = self.out(output1)
        aux_out = self.aux_out(output1)

        return torch.cat([out, aux_out], 1)