from transformers import BertModel
import torch.nn as nn
import torch


def log_sum_exp_mat(log_m, axis=-1):
    return torch.max(log_m, axis)[0] + torch.log(torch.exp(log_m - torch.max(log_m, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_tensor, axis=-1):
    return torch.max(log_tensor, axis)[0] + torch.log(
        torch.exp(log_tensor - torch.max(log_tensor, axis)[0].view(log_tensor.shape[0], -1, 1)).sum(axis))


class BertCRFTokenClassModel(nn.Module):

    def __init__(self, bert_path,
                 start_label_id=21, stop_label_id=22, num_labels=23):
        super(BertCRFTokenClassModel, self).__init__()

        self.bertModel = BertModel.from_pretrained(bert_path)

        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.hidden2label = nn.Linear(768, self.num_labels)
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

    def get_bert_features(self, input_ids, input_mask):

        bert_seq_out = self.bertModel(input_ids=input_ids,
                                      attention_mask=input_mask)
        bert_seq_out = bert_seq_out.last_hidden_state
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

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

    def neg_log_likelihood(self, input_ids, input_mask, label_ids):
        bert_feats = self.get_bert_features(input_ids, input_mask)
        forward_score = self.forward_alg(bert_feats)
        gold_score = self.score_sentence(bert_feats, label_ids)
        return torch.mean(forward_score - gold_score)

    def forward(self, input_ids, input_mask):

        bert_feats = self.get_bert_features(input_ids, input_mask)
        score, label_seq_ids = self.viterbi_decode(bert_feats)
        return score, label_seq_ids


if __name__ == '__main__':
    bert_path_ = r'../dataset/bert_chinese'
    bert_ids_ = torch.tensor([[345, 232, 13, 544, 2323]]).cuda()
    bert_mask_ = torch.tensor([[1, 1, 1, 1, 1]]).cuda()
    bert_model = BertCRFTokenClassModel(bert_path_).cuda()
    _, predict = bert_model.forward(bert_ids_, bert_mask_)
    print(predict.shape, predict)
