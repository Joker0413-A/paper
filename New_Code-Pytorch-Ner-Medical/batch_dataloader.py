import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn.functional as F

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
        return {'O': 0, 'B-dis': 1, 'I-dis': 2, 'B-pro': 3, 'I-pro': 4, 'B-dru': 5,
                'I-dru': 6, 'B-bod': 7, 'I-bod': 8, 'B-sym': 9, 'I-sym': 10,
                'B-ite': 11, 'I-ite': 12, 'B-dep': 13, 'I-dep': 14, 'B-mic': 15,
                'I-mic': 16, 'B-equ': 17, 'I-equ': 18, '[CLS]': 19, '[SEP]': 20}

    def __getitem__(self, index):
        item = self.data[index]
        #bert_ids = torch.tensor(item['input_ids'], dtype=torch.long).squeeze()
        #bert_mask = torch.tensor(item['attention_mask'], dtype=torch.long).squeeze()
        bert_ids = item['input_ids'].clone().detach().long().squeeze()
        bert_mask = item['attention_mask'].clone().detach().long().squeeze()

        rnn_array = torch.tensor(item['rnn_array'], dtype=torch.float)
        ner_labels = torch.tensor([self.label_to_index[label] for label in item['ner_labels']], dtype=torch.long)
        predict_mask = torch.tensor(item['predict_mask'], dtype=torch.bool).squeeze()

        return bert_ids, bert_mask, predict_mask, rnn_array, ner_labels

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    #fixed_len = max(x[0].shape[1] for x in batch)
    fixed_len = 200  # 固定的填充长度
    #fixed_len = 64  # 固定的填充长度
    padded_inputs, masks, predict_masks, lstm_arrays, labels = [], [], [], [], []

    for input_ids, attention_mask, predict_mask, rnn_array, ner_labels in batch:
        # 确保 input_ids 是二维的，进行填充并删除多余的维度
        input_ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
        input_ids = F.pad(input_ids, (0, fixed_len - input_ids.shape[-1]), value=0).squeeze(0)
        padded_inputs.append(input_ids)

        # 确保 attention_mask 是二维的，进行填充并删除多余的维度
        attention_mask = attention_mask if attention_mask.dim() == 2 else attention_mask.unsqueeze(0)
        attention_mask = F.pad(attention_mask, (0, fixed_len - attention_mask.shape[-1]), value=0).squeeze(0)
        masks.append(attention_mask)

        # 确保 predict_mask 是二维的，进行填充并删除多余的维度
        predict_mask = predict_mask if predict_mask.dim() == 2 else predict_mask.unsqueeze(0)
        predict_mask = F.pad(predict_mask, (0, fixed_len - predict_mask.shape[-1]), value=0).squeeze(0)
        predict_masks.append(predict_mask)

        # 确保 rnn_array 是三维的，进行填充并删除多余的维度
        rnn_array = rnn_array if rnn_array.dim() == 3 else rnn_array.unsqueeze(0)
        rnn_array = F.pad(rnn_array, (0, 0, 0, fixed_len - rnn_array.shape[-2]), value=0)
        # 对最后一个维度求均值，将维度从 [10, 1, 64, 200] 压缩到 [10, 64]
        rnn_array = rnn_array.mean(dim=-1).squeeze(1)
        # 确保 rnn_array 是二维的，进行填充并删除多余的维度
        rnn_array = rnn_array if rnn_array.dim() == 2 else rnn_array.unsqueeze(0)
        rnn_array = F.pad(rnn_array, (0, fixed_len - rnn_array.shape[-1]), value=0).squeeze(0)

        lstm_arrays.append(rnn_array)

        # 确保 ner_labels 是一维的，进行填充并删除多余的维度
        ner_labels = ner_labels if ner_labels.dim() == 1 else ner_labels.squeeze()
        ner_labels = F.pad(ner_labels, (0, fixed_len - ner_labels.shape[0]), value=0)
        labels.append(ner_labels)

    # 将列表转换为张量
    return torch.stack(padded_inputs), torch.stack(masks), torch.stack(predict_masks), torch.stack(lstm_arrays), torch.stack(labels)

if __name__ == "__main__":
    dataset = BatchLoaderData('E:/paper_code/paper/project2/pickle_data/train_data.pickle')
    loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)

    for inputs, masks, predict_masks, rnn_arrays, label_stack in loader:
        print(inputs.shape, masks.shape, predict_masks.shape, rnn_arrays.shape, label_stack.shape)


