import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, max_features, LSTM_UNITS, DENSE_HIDDEN_UNITS):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear_add_fts = nn.Linear(11, 64)
        self.dropout_add_fts = nn.Dropout(0.1)

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.bn = nn.BatchNorm1d(DENSE_HIDDEN_UNITS, momentum=0.5)
        self.dropout = nn.Dropout(0.1)

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

    def forward(self, x, lengths=None, f=None):
        # import pdb
        # pdb.set_trace()
        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)

        # print(h_embedding.shape)
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(h_embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        add_ft_branches = F.relu(self.linear_add_fts(f))
        add_ft_branches = self.dropout_add_fts(add_ft_branches)

        h_conc = torch.cat((max_pool, avg_pool, add_ft_branches), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc_linear1))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        hidden = self.bn(hidden)
        hidden = self.dropout(hidden)

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out