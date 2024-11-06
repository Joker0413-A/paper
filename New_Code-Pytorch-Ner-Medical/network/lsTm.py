import torch
import torch.nn as nn
from typing import Optional


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

        self.fc = nn.Linear(128, 23)

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


if __name__ == '__main__':

    data_ = torch.rand(1, 32, 200)
    model = LstModel()
    oust = model(data_)
    print(oust.shape)
    x, predicted = torch.max(oust, -1)
    print(predicted)

