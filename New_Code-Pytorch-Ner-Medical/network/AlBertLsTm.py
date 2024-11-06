import torch
import torch.nn as nn
from typing import Optional
from transformers import AlbertModel
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
                 input_size: Optional[int] = 768,
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
        x = self.fc(x)
        return x


class AlBertLsTmTokenClassModel(nn.Module):

    def __init__(self, albert_path):
        super(AlBertLsTmTokenClassModel, self).__init__()
        self.bertModel = AlbertModel.from_pretrained(albert_path)
        self.lstmModel = LstModel()

    def forward(self, bert_ids, bert_mask):
        x = self.bertModel(input_ids=bert_ids, attention_mask=bert_mask)
        input_lstm = x.last_hidden_state
        x = self.lstmModel(input_lstm)
        return x


if __name__ == '__main__':
    bert_path_ = r'../albert_-chinese-base'
    bert_ids_ = torch.tensor([[345, 232, 13, 544, 2323]])
    bert_mask_ = torch.tensor([[1, 1, 1, 1, 1]])
    model = AlBertLsTmTokenClassModel(bert_path_)
    bert_out = model(bert_ids_, bert_mask_)
    print(bert_out, bert_out.shape)
