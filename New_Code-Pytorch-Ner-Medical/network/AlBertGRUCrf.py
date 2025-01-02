import torch
import torch.nn as nn
from transformers import AlbertModel
from torchcrf import CRF
import gensim


# 加载Word2Vec模型
class Word2VecModel:
    def __init__(self, word2vec_file):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False, encoding="utf-8")

    def get_word_vector(self, word):
        # 如果word在词向量表中，则返回对应的词向量，否则返回全零向量
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return [0] * 200  # 假设Word2Vec向量的维度为200


# 归一化层
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


# GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size=768 + 200, hidden_size=128, num_layers=1, bidirectional=True):  # 输入维度调整为拼接后的尺寸
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional)
        self.layer_norm = LayerNormal(hidden_size * 2)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        #x = self.dropout(x)
        return x


# ALBERT + GRU + CRF 模型
class AlBertGRUCRFTokenClassModel(nn.Module):
    def __init__(self, albert_path, word2vec_file, num_labels=21, start_label_id=19, stop_label_id=20):
        super(AlBertGRUCRFTokenClassModel, self).__init__()
        self.albert = AlbertModel.from_pretrained(albert_path)
        self.word2vec_model = Word2VecModel(word2vec_file)  # 加载Word2Vec模型
        self.gru = GRUModel()
        self.hidden2label = nn.Linear(256, num_labels)  # 注意：这里的输入维度可能需要调整，根据拼接后的向量
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))

        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.stop_label_id] = -10000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 添加用于计算注意力的线性层
        self.attention_layer = nn.Linear(768 + 200, 1)  # ALBERT和Word2Vec拼接后的维度

    def get_albert_features(self, input_ids, attention_mask):
        # 1) 获取 ALBERT 输出
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # 2) 准备获取 Word2Vec 向量
        batch_size, seq_len = input_ids.size()
        all_word_vecs = []

        # 遍历 batch 内所有样本
        for b_idx in range(batch_size):
            single_sample_word_vecs = []
            # 遍历一个样本内的所有 token
            for i in range(seq_len):
                token_id = input_ids[b_idx][i].item()
                # 这里 token_id 直接做 key 去查 Word2Vec
                # 如果你的 Word2VecModel 是以 "词(字符串)"为 key，而不是 "id"，
                # 需要先把 id -> token 映射到对应的文本词，再去查 word2vec
                word_vec = self.word2vec_model.get_word_vector(token_id)
                single_sample_word_vecs.append(word_vec)

            # [seq_len, 200] 转成 Torch tensor
            single_sample_word_vecs = torch.tensor(single_sample_word_vecs,
                                                   dtype=torch.float,
                                                   device=sequence_output.device)
            all_word_vecs.append(single_sample_word_vecs)

        # 拼接成 [batch_size, seq_len, 200]
        word_vec_tensor = torch.stack(all_word_vecs, dim=0)

        # 3) 拼接 ALBERT 输出和 Word2Vec 向量 => [batch_size, seq_len, 768 + 200]
        combined_output = torch.cat((sequence_output, word_vec_tensor), dim=-1)

        # 4) 注意力计算
        attention_scores = self.attention_layer(combined_output)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = combined_output * attention_weights  # [batch_size, seq_len, 968]

        # 5) GRU -> Linear => emissions
        gru_output = self.gru(weighted_output)  # [batch_size, seq_len, 256]
        emissions = self.hidden2label(gru_output)  # [batch_size, seq_len, num_labels]

        return emissions

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
        predictions_padded = [torch.tensor(p + [self.stop_label_id] * (200 - len(p)), dtype=torch.long) for p in
                              predictions]
        predictions_tensor = torch.stack(predictions_padded).to(self.device)
        score = torch.zeros(predictions_tensor.size(0), dtype=torch.float, device=self.device)  # 占位符，实际可能是一些有用的评分
        return score, predictions_tensor  # 返回占位分数和预测张量


# Example usage
if __name__ == "__main__":
    albert_path = r'E:/paper_code/paper/project2/New_Code-Pytorch-Ner-Medical/dataset/albert_chinese'
    word2vec_file = r'E:/paper_code/paper/project2/New_Code-Pytorch-Ner-Medical/dataset/word_vce_gru.txt'  # 替换为正确的Word2Vec文件路径
    model = AlBertGRUCRFTokenClassModel(albert_path, word2vec_file)
    input_ids = torch.randint(0, 1000, (1, 10))  # 示例输入
    attention_mask = torch.ones((1, 10))
    predictions = model(input_ids, attention_mask)
    print("Predictions:", predictions)
