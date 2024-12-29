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
        # 获取ALBERT的输出
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # ALBERT的输出，默认在cuda上

        # 使用Word2Vec为每个词获取词向量
        word2vec_features = []
        word_vec_tensor = []

        for i in range(sequence_output.size(1)):  # 对每个词进行操作
            word = input_ids[0][i].item()  # 获取词的ID
            word_vec = self.word2vec_model.get_word_vector(word)  # 查找词向量
            word_vec_tensor.append(word_vec)

        # 将word_vec_tensor转换为batch_size, seq_len, word_vector_size的形状
        word_vec_tensor = torch.tensor(word_vec_tensor, dtype=torch.float).to(
            sequence_output.device)  # 转为tensor并保证在相同设备
        word_vec_tensor = word_vec_tensor.unsqueeze(0).expand(sequence_output.size(0), -1,
                                                              -1)  # 扩展为 (batch_size, seq_len, word_vector_size)

        # 拼接ALBERT的输出和Word2Vec的向量
        combined_output = torch.cat((sequence_output, word_vec_tensor), dim=-1)  # 拼接在最后一维

        # 计算注意力权重
        attention_scores = self.attention_layer(combined_output)  # shape: [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # 计算softmax来得到每个词的注意力权重

        # 使用注意力权重对特征进行加权
        weighted_output = combined_output * attention_weights  # [batch_size, seq_len, hidden_size + 200]

        # 将加权后的特征传递给GRU层
        gru_output = self.gru(weighted_output)
        emissions = self.hidden2label(gru_output)
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
