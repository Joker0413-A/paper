import numpy as np
import torch
from transformers import BertTokenizer
from tools import save_pickle
from textrank4zh import TextRank4Sentence
from gensim.models import KeyedVectors
from collections import Counter
import os

class DataEncodePrecess(object):
    def __init__(self, tokenizer_file, word2vce_file):
        super(DataEncodePrecess, self).__init__()

        # 加载本地 BERT 模型
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
        self.word2vce = KeyedVectors.load_word2vec_format(word2vce_file,
                                                          binary=False,
                                                          encoding="utf-8",
                                                          unicode_errors='ignore')

    def bert_encode(self, bert_text):
        words = self.tokenizer.encode_plus(bert_text, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return {"input_ids": words['input_ids'], "attention_mask": words['attention_mask']}

    def rank_encode(self, lstm_text):
        result = ''
        for x in lstm_text:
            result += x
        tr4s = TextRank4Sentence()
        tr4s.analyze(result, lower=True, source='all_filters')
        sentences = tr4s.get_key_sentences(num=1, sentence_min_len=1)
        sentences = [x for x in sentences[0]['sentence']]
        key_sentence = self.lstm_encode(lstm_text=sentences)
        return key_sentence

    def lstm_encode(self, lstm_text):
        words = []
        for word in lstm_text:
            if word in self.word2vce:
                words.append(self.word2vce[word].tolist())
            else:
                words.append([0] * 200)
        return words


def read_data_save_pickle(input_file, tokenizer_file, word2vce_file):
    feature_encode = DataEncodePrecess(
        tokenizer_file=tokenizer_file,
        word2vce_file=word2vce_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        out_lists = []
        entries = f.read().strip().split("\n\n")
        all_labels = []
        for entry in entries:
            words = []
            ner_labels = []

            for line in entry.splitlines():
                pieces = line.strip().split()
                if len(pieces) < 2:
                    continue
                words.append(pieces[0])
                ner_labels.append(pieces[-1])

            bert_encode_dict = feature_encode.bert_encode(bert_text=words)
            lstm_encode = feature_encode.lstm_encode(lstm_text=words)
            rank_encode = feature_encode.rank_encode(lstm_text=words)

            bert_encode_dict['rnn_array'] = lstm_encode
            # bert_encode_dict['rank_array'] = rank_encode

            predict_mask = [0] + [1] * len(ner_labels)

            bert_encode_dict['ner_labels'] = ['[CLS]'] + ner_labels + ['[SEP]']
            bert_encode_dict['predict_mask'] = predict_mask
            out_lists.append(bert_encode_dict)
            all_labels += ner_labels
    return out_lists, all_labels


def save():
    bert_model_path = r'E:\paper_code\paper\project2\New_Code-Pytorch-Ner-Medical\dataset\bert_chinese'
    wordvce = r'E:\paper_code\paper\project2\New_Code-Pytorch-Ner-Medical\dataset\word_vce.txt'

    data_list_train, all_labels_train = read_data_save_pickle(
        input_file=r'E:\paper_code\paper\project2\New_Code-Pytorch-Ner-Medical\dataset\CMeEE-V2_train_split.txt',
        tokenizer_file=bert_model_path,  # 使用绝对路径
        word2vce_file=wordvce
    )
    save_pickle(data_list_train, '../../pickle_data/train_data.pickle')

    data_list_valid, all_labels_valid = read_data_save_pickle(
        input_file=r'E:\paper_code\paper\project2\New_Code-Pytorch-Ner-Medical\dataset\CMeEE-V2_dev_split.txt',
        tokenizer_file=bert_model_path,  # 正确路径
        word2vce_file=wordvce
    )
    save_pickle(data_list_valid, '../../pickle_data/train_valid.pickle')

    data_list_test, all_labels_test = read_data_save_pickle(
        input_file=r'E:\paper_code\paper\project2\New_Code-Pytorch-Ner-Medical\dataset\CMeEE-V2_test_split.txt',
        tokenizer_file=bert_model_path,  # 正确路径
        word2vce_file=wordvce
    )
    save_pickle(data_list_test, '../../pickle_data/train_test.pickle')

    print(f'Train Set Labels {Counter(all_labels_train)}')
    print(f'Valid Set Labels {Counter(all_labels_valid)}')
    print(f'Test Set Labels {Counter(all_labels_test)}')



if __name__ == '__main__':
    save()
