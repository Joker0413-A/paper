from transformers import BertModel
import torch.nn as nn
import torch


class BertTokenClassModel(nn.Module):

    def __init__(self, bert_path):
        super(BertTokenClassModel, self).__init__()
        self.bertModel = BertModel.from_pretrained(bert_path)

        self.fc = nn.Linear(768, 23)

    def forward(self, bert_ids, bert_mask):
        x = self.bertModel(input_ids=bert_ids, attention_mask=bert_mask)
        x = x.last_hidden_state
        x = self.fc(x)
        return x


if __name__ == '__main__':
    bert_path_ = r'../dataset/bert_chinese'
    bert_ids_ = torch.tensor([[345, 232, 13, 544, 2323]])
    bert_mask_ = torch.tensor([[1, 1, 1, 1, 1]])
    bert_model = BertTokenClassModel(bert_path_)
    bert_out = bert_model(bert_ids_, bert_mask_)
    print(bert_out.shape)
