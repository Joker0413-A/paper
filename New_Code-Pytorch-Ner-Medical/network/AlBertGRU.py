import torch
import torch.nn as nn
from transformers import AlbertModel

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
    def __init__(self,
                 input_size: int = 768,
                 hidden_size: int = 128,
                 num_layers: int = 1
                 ):
        super(GRUModel, self).__init__()
        self.sen_rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=False)
        self.layer_norm = LayerNormal(hidden_size)
        self.fc = nn.Linear(hidden_size, 21)

    def forward(self, x):
        x, _ = self.sen_rnn(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x

class AlBertGRUTokenClassModel(nn.Module):
    def __init__(self, albert_path):
        super(AlBertGRUTokenClassModel, self).__init__()
        self.albertModel = AlbertModel.from_pretrained(albert_path)
        self.gruModel = GRUModel()

    def forward(self, bert_ids, bert_mask):
        x = self.albertModel(input_ids=bert_ids, attention_mask=bert_mask)
        input_gru = x.last_hidden_state
        x = self.gruModel(input_gru)
        return x

# Example usage
if __name__ == '__main__':
    albert_path = r'E:/paper_code/paper/project2/New_Code-Pytorch-Ner-Medical/dataset/albert_chinese'
    bert_ids = torch.tensor([[345, 232, 13, 544, 2323]])
    bert_mask = torch.tensor([[1, 1, 1, 1, 1]])
    model = AlBertGRUTokenClassModel(albert_path)
    output = model(bert_ids, bert_mask)
    print(output, output.shape)
