import torch
import torch.nn as nn
from transformers import AlbertModel
from torchcrf import CRF

class LayerNormal(nn.Module):
    def __init__(self, hidden_size, esp=1e-6):
        super(LayerNormal, self).__init__()
        self.esp = esp
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True).clamp(min=self.esp)
        out = (x - mu) / sigma
        out = out * self.weight.expand_as(out) + self.bias.expand_as(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=1, bidirectional=True):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.layer_norm = LayerNormal(hidden_size * 2)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        return x

class AlBertGRUCRFTokenClassModel(nn.Module):
    def __init__(self, albert_path, num_labels=21, start_label_id=19, stop_label_id=20):
        super(AlBertGRUCRFTokenClassModel, self).__init__()
        self.albert = AlbertModel.from_pretrained(albert_path)
        self.gru = GRUModel()
        self.hidden2label = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.stop_label_id] = -10000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_albert_features(self, input_ids, attention_mask):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        gru_output = self.gru(sequence_output)
        emissions = self.hidden2label(gru_output)
        return emissions

    def forward_alg(self, feats):
        # Implementation for the forward algorithm
        pass

    def viterbi_decode(self, feats, mask):
        predictions = self.crf.decode(feats, mask=mask.byte())
        predictions_padded = [torch.tensor(p + [-1] * (200 - len(p)), dtype=torch.long) for p in predictions]
        predictions_tensor = torch.stack(predictions_padded).to(self.device)
        return predictions_tensor

    def neg_log_likelihood(self, input_ids, input_mask, label_ids):
        feats = self.get_albert_features(input_ids, input_mask)
        return -self.crf(feats, label_ids, mask=input_mask.byte())

    def forward(self, input_ids, input_mask):
        feats = self.get_albert_features(input_ids, input_mask)
        mask = input_mask != 0
        predictions = self.crf.decode(feats, mask=mask.byte())
        # 如有需要，调整以下代码以确保预测长度固定，比如固定长度200
        predictions_padded = [torch.tensor(p + [self.stop_label_id] * (200 - len(p)), dtype=torch.long) for p in predictions]
        predictions_tensor = torch.stack(predictions_padded).to(self.device)
        score = torch.zeros(predictions_tensor.size(0), dtype=torch.float, device=self.device)  # 占位符，实际可能是一些有用的评分
        return score, predictions_tensor  # 返回占位分数和预测张量

# Example usage
if __name__ == "__main__":
    albert_path = r'../albert_chinese'
    model = AlBertGRUCRFTokenClassModel(albert_path)
    input_ids = torch.randint(0, 1000, (1, 10))
    attention_mask = torch.ones((1, 10))
    predictions = model(input_ids, attention_mask)
    print("Predictions:", predictions)
