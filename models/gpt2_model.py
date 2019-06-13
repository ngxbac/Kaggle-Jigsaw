import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model


class GPT2ClassificationMultioutput(GPT2PreTrainedModel):

    def __init__(self, config, clf_dropout=0.4, num_aux_labels=6):
        super(GPT2ClassificationMultioutput, self).__init__(config)
        n_hidden = config.n_embd * 2 + 64

        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(clf_dropout)

        self.linear_add_fts = nn.Linear(11, 64)
        self.dropout_add_fts = nn.Dropout(0.1)

        self.linear_target = nn.Linear(n_hidden, 1)
        self.linear_aux_target = nn.Linear(n_hidden, num_aux_labels)

        nn.init.normal_(self.linear_target.weight, std=0.02)
        nn.init.normal_(self.linear_target.bias, 0)

        nn.init.normal_(self.linear_aux_target.weight, std=0.02)
        nn.init.normal_(self.linear_aux_target.bias, 0)
        self.apply(self.init_weights)

    def forward(self, input_ids, features, position_ids=None, token_type_ids=None, lm_labels=None, past=None, **kwargs):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)

        add_ft_branches = F.relu(self.linear_add_fts(features))
        add_ft_branches = self.dropout_add_fts(add_ft_branches)

        h_conc = torch.cat((max_pool, avg_pool, add_ft_branches), 1)

        target = self.linear_target(h_conc)
        aux_target = self.linear_aux_target(h_conc)

        return torch.cat([target, aux_target], 1)
