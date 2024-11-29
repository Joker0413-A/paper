import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertGRUTokenClassModel(nn.Module):
    def __init__(self, bert_path, num_labels=21):
        super(BertGRUTokenClassModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_path)
        self.gru = nn.GRU(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, bert_ids, bert_mask):
        outputs = self.bert(input_ids=bert_ids, attention_mask=bert_mask)
        sequence_output = outputs[0]
        gru_output, _ = self.gru(sequence_output)
        logits = self.classifier(gru_output)
        return logits


if __name__ == "__main__":
    bert_path = r'../dataset/bert_chinese'  # 指向BERT模型
    bert_ids = torch.tensor([[345, 232, 13, 544, 2323]])  # 示例输入ID
    bert_mask = torch.tensor([[1, 1, 1, 1, 1]])  # 输入的注意力掩码

    # 实例化模型
    model = BertGRUTokenClassModel(bert_path=bert_path, num_labels=10)
    model.eval()  # 设置为评估模式


    with torch.no_grad():  # 关闭梯度计算
        bert_out = model(bert_ids, bert_mask)

    # 打印输出张量及其形状
    print("Output from model:", bert_out)
    print("Shape of output:", bert_out.shape)
