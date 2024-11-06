from transformers import AlbertModel, BertForTokenClassification
import torch.nn as nn
import torch


class AlBertTokenClassModel(nn.Module):

    def __init__(self, albert_path):
        super(AlBertTokenClassModel, self).__init__()
        self.bertModel = AlbertModel.from_pretrained(albert_path)

        self.fc = nn.Linear(768, 23)

    def forward(self, bert_ids, bert_mask):
        x = self.bertModel(input_ids=bert_ids, attention_mask=bert_mask)
        x = x.last_hidden_state
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # # loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    fun = nn.CrossEntropyLoss()
    bert_path_ = r'../albert_-chinese-base'
    bert_ids_ = torch.LongTensor([[345, 232, 13, 544, 2323]])
    bert_mask_ = torch.LongTensor([[1, 1, 1, 1, 1]])
    label = torch.LongTensor([[0, 1, 2, 3, 4]])
    bert_model = AlBertTokenClassModel(bert_path_)
    bert_out = bert_model(bert_ids_, bert_mask_)
    print(label.shape)
    print(bert_out.shape)
    print(label.view(-1).shape)
    print(bert_out.view(-1, 23).shape)
    x, y = torch.max(bert_out, -1)
    print(x, y)
    loss = fun(bert_out.view(-1, 23), label.view(-1))
    print(loss)
