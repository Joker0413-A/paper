import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence

class BatchLoaderData(Dataset):
    def __init__(self, pickle_file):
        super(BatchLoaderData, self).__init__()
        self.data = self.load_pickle(pickle_file)
        self.label_to_index = self.generate_label_to_index()

    def load_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as open_file:
            pickle_data = pickle.load(open_file)
        return pickle_data

    def generate_label_to_index(self):
        # 标签映射
        return { 'O': 0, 'B-dis': 1, 'I-dis': 2, 'B-pro': 3, 'I-pro': 4, 'B-dru': 5,
            'I-dru': 6, 'B-bod': 7, 'I-bod': 8, 'B-sym': 9, 'I-sym': 10,
            'B-ite': 11, 'I-ite': 12, 'B-dep': 13, 'I-dep': 14, 'B-mic': 15,
            'I-mic': 16, 'B-equ': 17, 'I-equ': 18, '[CLS]': 19, '[SEP]': 20
        }

    def __getitem__(self, index):
        item = self.data[index]
        bert_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        bert_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        rnn_array = torch.tensor(item['rnn_array'], dtype=torch.float)
        ner_labels = torch.tensor([self.label_to_index[label] for label in item['ner_labels']], dtype=torch.long)
        predict_mask = torch.tensor(item['predict_mask'], dtype=torch.bool)

        return bert_ids, bert_mask, predict_mask, rnn_array, ner_labels

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    max_len = max(max(x[0].shape[1], x[3].shape[0]) for x in batch)  # 找到批次中 input_ids 和 rnn_array 的最大长度
    max_label_len = max(len(x[4]) for x in batch)  # 找到批次中 ner_labels 的最大长度

    padded_inputs, masks, predict_masks, lstm_arrays, labels = [], [], [], [], []

    for input_ids, attention_mask, predict_mask, rnn_array, ner_labels in batch:
        # 计算需要填充的长度
        padding_length = max_len - input_ids.shape[1]

        # 填充 input_ids
        padded_input_ids = torch.cat([
            input_ids,
            torch.zeros((1, padding_length), dtype=input_ids.dtype)
        ], dim=1)
        padded_inputs.append(padded_input_ids)

        # 填充 attention_mask
        padded_attention_mask = torch.cat([
            attention_mask,
            torch.zeros((1, padding_length), dtype=attention_mask.dtype)
        ], dim=1)
        masks.append(padded_attention_mask)

        # 填充 predict_mask
        if predict_mask.dim() == 1:
            predict_mask = predict_mask.unsqueeze(0)
        padded_predict_mask = torch.cat([
            predict_mask,
            torch.zeros((1, padding_length), dtype=predict_mask.dtype)
        ], dim=1)
        predict_masks.append(padded_predict_mask)

        # 填充 rnn_array 到 max_len
        rnn_padding_length = max_len - rnn_array.shape[0]
        if rnn_padding_length > 0:
            padded_lstm_array = torch.cat([
                rnn_array,
                torch.zeros((rnn_padding_length, rnn_array.shape[1]), dtype=rnn_array.dtype)
            ], dim=0)
        else:
            padded_lstm_array = rnn_array
        lstm_arrays.append(padded_lstm_array)

        # 填充 ner_labels 到 max_label_len
        padded_labels = torch.cat([
            ner_labels,
            torch.zeros((max_label_len - len(ner_labels),), dtype=ner_labels.dtype)
        ])
        labels.append(padded_labels)

    # 使用 torch.stack 将列表转换为张量
    return torch.stack(padded_inputs), torch.stack(masks), torch.stack(predict_masks), torch.stack(lstm_arrays), torch.stack(labels)



if __name__ == "__main__":
    dataset = BatchLoaderData('E:/paper_code/paper/project2/pickle_data/train_data.pickle')
    loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)

    for inputs, masks, predict_masks, rnn_arrays, label_stack in loader:
        print(inputs.shape, masks.shape, predict_masks.shape, rnn_arrays.shape, label_stack.shape)

