import torch
import torch.nn as nn
from typing import Optional
import torch


class LayerNormal(nn.Module):
    def __init__(self, hidden_size, esp=1e-6):
        super(LayerNormal, self).__init__()
        self.esp = esp
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = torch.mean(input=x, dim=-1, keepdim=True)
        sigma = torch.std(input=x, dim=-1, keepdim=True).clamp(min=self.esp)
        out = (x - mu) / sigma
        out = out * self.weight.expand_as(out) + self.bias.expand_as(out)
        return out


class LstModel(nn.Module):
    def __init__(self,
                 input_size: Optional[int] = 200,
                 hidden_size: Optional[int] = 128,
                 num_layers: Optional[int] = 1
                 ):
        super(LstModel, self).__init__()

        self.sen_rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=False)

        self.LayerNormal = LayerNormal(hidden_size)

        self.fc = nn.Linear(hidden_size, 23)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):

        x, _ = self.sen_rnn(x, None)
        x = self.LayerNormal(x)
        return x


def log_sum_exp_mat(log_m, axis=-1):
    return torch.max(log_m, axis)[0] + torch.log(torch.exp(log_m - torch.max(log_m, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_tensor, axis=-1):
    return torch.max(log_tensor, axis)[0] + torch.log(
        torch.exp(log_tensor - torch.max(log_tensor, axis)[0].view(log_tensor.shape[0], -1, 1)).sum(axis))


class LsTmCRF(nn.Module):

    def __init__(self, start_label_id=21, stop_label_id=22, num_labels=23):
        super(LsTmCRF, self).__init__()

        self.lstm = LstModel()

        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels

        self.hidden2label = nn.Linear(128, self.num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def forward_alg(self, feats):

        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_alpha[:, 0, self.start_label_id] = 0

        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        log_prob_all_bar = log_sum_exp_batch(log_alpha)

        return log_prob_all_bar

    def get_rnn_features(self, rnn_array):
        rnn_seq_out = self.lstm(rnn_array)
        rnn_feats = self.hidden2label(rnn_seq_out)
        return rnn_feats

    def score_sentence(self, feats, label_ids):

        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)

        for t in range(1, T):
            score = score + \
                    batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t - 1]).view(-1, 1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def viterbi_decode(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)

        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)
        max_log_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_log_allz_allx, path

    def neg_log_likelihood(self, rnn_array, label_ids):
        bert_feats = self.get_rnn_features(rnn_array)
        forward_score = self.forward_alg(bert_feats)
        gold_score = self.score_sentence(bert_feats, label_ids)
        return torch.mean(forward_score - gold_score)

    def forward(self, rnn_array):
        feats = self.get_rnn_features(rnn_array)
        score, label_seq_ids = self.viterbi_decode(feats)
        return score, label_seq_ids


if __name__ == '__main__':
    rnn = torch.rand(2, 32, 200).cuda()
    model = LsTmCRF().cuda()
    _, predict = model(rnn)
    print(predict, predict.shape)

