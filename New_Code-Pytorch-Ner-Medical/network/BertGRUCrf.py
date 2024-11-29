import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from torchcrf import CRF

class BertGRUCrfTokenClassModel(nn.Module):
    def __init__(self, bert_path, num_labels=21):
        super(BertGRUCrfTokenClassModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.gru = nn.GRU(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2label = nn.Linear(256, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(self, input_ids, input_mask):
        emissions = self._get_gru_features(input_ids, input_mask)
        predictions = self.crf.decode(emissions, mask=input_mask.byte())
        # 手动截断或填充predictions到长度为200
        predictions_padded = [torch.tensor(p[:200] if len(p) > 200 else p + [-1] * (200 - len(p)), dtype=torch.long) for p in predictions]
        predictions_tensor = pad_sequence(predictions_padded, batch_first=True, padding_value=-1).to(input_mask.device)
        score = self._score(emissions, predictions_tensor, input_mask)  # Optional: Calculate the score of the predictions
        return score, predictions_tensor

    def neg_log_likelihood(self, input_ids, input_mask, label_ids):
        emissions = self._get_gru_features(input_ids, input_mask)
        loss = -self.crf(emissions, label_ids, mask=input_mask.byte())
        return loss

    def _get_gru_features(self, input_ids, input_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=input_mask)
        sequence_output = outputs.last_hidden_state
        gru_output, _ = self.gru(sequence_output)
        emissions = self.hidden2label(gru_output)
        return emissions

    def _score(self, emissions, tags, mask):
        # 实际得分计算逻辑可以根据需要实现
        return torch.tensor(0.0, device=emissions.device)  # 返回具有适当设备的张量

# Example usage:
if __name__ == "__main__":
    bert_path = r'../dataset/bert_chinese'
    num_labels = 21
    model = BertGRUCrfTokenClassModel(bert_path=bert_path, num_labels=num_labels)
    input_ids = torch.randint(0, 1000, (1, 10))  # Example input
    attention_mask = torch.ones((1, 10))
    score, predictions = model(input_ids, attention_mask)
    print("Score:", score)
    print("Predictions:", predictions)
