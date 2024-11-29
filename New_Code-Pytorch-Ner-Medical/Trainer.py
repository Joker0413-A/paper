import torch.optim
import scipy.io as scio
from tools import *
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
import numpy as np
from sklearn.metrics import classification_report
import os

from network.AlBert import AlBertTokenClassModel
from network.AlBertCRF import AlBertCRFTokenClassModel
from network.AlBertLsTm import AlBertLsTmTokenClassModel
from network.AlBertLsTmCRF import AlBertLsTmCRFTokenClassModel
from network.AlBertFusionAttCRF import AlBertFusionAttCRFTokenClassModel

from network.Bert import BertTokenClassModel
from network.BertCRF import BertCRFTokenClassModel
from network.BertLsTm import BertLsTmTokenClassModel
from network.BertLsTmCRF import BertLsTmCRFTokenClassModel
from network.BertFusionAttCRF import BertFusionAttCRFTokenClassModel

from network.lsTm import LstModel
from network.lsTmCRF import LsTmCRF

from network.BertGRU import BertGRUTokenClassModel  # 导入 BertGRU 模型类
from network.BertGRUCrf import BertGRUCrfTokenClassModel
from network.AlBertGRU import AlBertGRUTokenClassModel
from network.AlBertGRUCrf import AlBertGRUCRFTokenClassModel


from batch_dataloader import BatchLoaderData, collate_fn
import torch.nn as nn

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Trainer(object):
    def __init__(self,
                 model_chose,
                 batch_size,
                 epochs
                 ):

        super(Trainer, self).__init__()
        # 初始化最佳指标的变量
        self.best_acc = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_f1 = 0

        chose = ['Bert', 'BertCrf', 'BertLstM', 'BertLstMCrf', 'BertFusionAttCrf',
                 'AlBert', 'AlBertCrf', 'AlBertLstM', 'AlBertLstMCrf', 'AlBertFusionAttCrf',
                 'Lstm', 'LsTmCrf','BertGRU', 'BertGRUCrf','AlBertGRU','AlBertGRUCrf']

        if model_chose not in chose:
            raise NameError("Model_combination should be one of {}, But you have chosen '{}', please correct it".
                            format(chose, model_chose))

        self.num_labels = 21
        self.batch_size = batch_size
        self.clip = -1
        self.epochs = epochs
        self.model_chose = model_chose

        bert_config = r'E:/paper_code/paper/project2/New_Code-Pytorch-Ner-Medical/dataset/bert_chinese'
        albert_config = r'E:/paper_code/paper/project2/New_Code-Pytorch-Ner-Medical/dataset/albert_chinese'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if model_chose == 'Bert':
            model = BertTokenClassModel(bert_path=bert_config)
            self.optimizer = self.get_optimizer(model, lr=2e-5)
        if model_chose == 'BertCrf':
            model = BertCRFTokenClassModel(bert_path=bert_config)
            self.optimizer = self.get_optimizer(model, lr=1e-4)
        if model_chose == 'BertLstM':
            model = BertLsTmTokenClassModel(bert_path=bert_config)
            self.optimizer = self.get_optimizer(model, lr=2e-5)
        if model_chose == 'BertLstMCrf':
            model = BertLsTmCRFTokenClassModel(bert_path=bert_config)
            self.optimizer = self.get_optimizer(model, lr=1e-4)
        if model_chose == 'BertFusionAttCrf':
            model = BertFusionAttCRFTokenClassModel(bert_path=bert_config)
            self.optimizer = self.get_optimizer(model, lr=2e-5)

        if model_chose == 'AlBert':
            model = AlBertTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=2e-5)
        if model_chose == 'AlBertCrf':
            model = AlBertCRFTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=1e-4)
        if model_chose == 'AlBertLstM':
            model = AlBertLsTmTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=2e-5)
        if model_chose == 'AlBertLstMCrf':
            model = AlBertLsTmCRFTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=1e-4)
        if model_chose == 'AlBertFusionAttCrf':
            model = AlBertFusionAttCRFTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=1e-4)

        if model_chose == 'Lstm':
            model = LstModel()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        if model_chose == 'LsTmCrf':
            model = LsTmCRF()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if model_chose == 'BertGRU':
            model = BertGRUTokenClassModel(bert_path=bert_config)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 根据需要调整学习率

        if model_chose == 'BertGRUCrf':
            model = BertGRUCrfTokenClassModel(bert_path=bert_config)
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 根据需要调整学习率

        if model_chose == 'AlBertGRU':
            model = AlBertGRUTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=2e-5)

        if model_chose == 'AlBertGRUCrf':
            model = AlBertGRUCRFTokenClassModel(albert_path=albert_config)
            self.optimizer = self.get_optimizer(model, lr=1e-4)

        self.model = model.to(self.device)

    def train(self, epoch, train_loader, len_train_set):

        self.model.train()
        train_loss, train_acc, = 0, 0
        for batch_idx, batch in enumerate(train_loader):

            batch = [x.to(self.device) for x in batch]
            ids_list, mask_list, predict_list, lstm_array, label_list = \
                batch[0], batch[1], batch[2], batch[3], batch[4]
            # 打印 lstm_array 的形状
            #print("lstm_array shape in Trainer.train:", lstm_array.shape)

            if self.model_chose == 'Lstm':
                output = self.model.forward(x=lstm_array)
                loss_train = self.criterion(output.view(-1, self.num_labels), label_list.view(-1))

            if self.model_chose == 'LsTmCrf':
                loss_train = self.model.neg_log_likelihood(rnn_array=lstm_array,
                                                           label_ids=label_list)

            if self.model_chose in ['Bert', 'BertLstM', 'AlBert', 'AlBertLstM', 'BertGRU', 'AlBertGRU']:
                output = self.model.forward(bert_ids=ids_list,
                                            bert_mask=mask_list)
                loss_train = self.criterion(output.view(-1, self.num_labels), label_list.view(-1))

            if self.model_chose in ['BertCrf', 'BertLstMCrf', 'AlBertCrf', 'AlBertLstMCrf', 'BertGRUCrf', 'AlBertGRUCrf']:
                loss_train = self.model.neg_log_likelihood(input_ids=ids_list,
                                                           input_mask=mask_list,
                                                           label_ids=label_list)

            if self.model_chose in ['BertFusionAttCrf', 'AlBertFusionAttCrf']:
                loss_train = self.model.neg_log_likelihood(input_ids=ids_list,
                                                           input_mask=mask_list,
                                                           lstm_array=lstm_array,
                                                           label_ids=label_list)




            loss = loss_train
            self.optimizer.zero_grad()
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            loss.backward()
            self.optimizer.step()
            train_loss += loss * label_list.size(0)

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{}] \tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.batch_size, len_train_set, loss.item()))

        train_loss = torch.true_divide(train_loss, len_train_set)
        print('Train set: Average loss: {:.6f}'.format(train_loss))
        return train_loss

    def valid(self, epoch, valid_loader, test_loader, model_chose):

        predicts, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for valid_idx, valid_batch in enumerate(valid_loader):

                valid_batch = [x.to(self.device) for x in valid_batch]

                ids_list, mask_list, predict_list, lstm_array, label_list = \
                    valid_batch[0], valid_batch[1], valid_batch[2], valid_batch[3], valid_batch[4]

                if self.model_chose == 'Lstm':
                    out_scores = self.model.forward(x=lstm_array)
                    _, predicted = torch.max(out_scores, -1)

                if self.model_chose == 'LsTmCrf':
                    _, predicted = self.model.forward(rnn_array=lstm_array)

                if self.model_chose in ['Bert', 'BertLstM', 'AlBert', 'AlBertLstM', 'BertGRU', 'AlBertGRU']:
                    out_scores = self.model.forward(bert_ids=ids_list,
                                                    bert_mask=mask_list)
                    _, predicted = torch.max(out_scores, -1)

                if self.model_chose in ['BertCrf', 'BertLstMCrf', 'AlBertCrf', 'AlBertLstMCrf', 'BertGRUCrf', 'AlBertGRUCrf']:
                    _, predicted = self.model.forward(input_ids=ids_list, input_mask=mask_list)

                if self.model_chose in ['BertFusionAttCrf', 'AlBertFusionAttCrf']:
                    _, predicted = self.model.forward(input_ids=ids_list,
                                                      input_mask=mask_list,
                                                      lstm_array=lstm_array)

                # 确保预测和掩码在相同的设备上
                predicted = predicted.to(self.device)  # 将predicted张量移到设备上
                predict_list = predict_list.to(self.device)  # 确保predict_list也在同一设备

                predicted_list = torch.masked_select(predicted, predict_list)
                ture_ids = torch.masked_select(label_list, predict_list)

                predicts.extend(predicted_list.cpu().numpy().tolist())
                targets.extend(ture_ids.cpu().numpy().tolist())

            valid_accuracy = accuracy_score(predicts, targets)
            valid_precision = precision_score(targets, predicts, average='macro')
            valid_recall = recall_score(targets, predicts, average='macro')
            valid_f1 = f1_score(targets, predicts, average='macro')

            print('\nValid set: Accuracy: ({:.5f}), Best_Accuracy({:.5f})'.format(valid_accuracy, best_acc))
            if valid_accuracy > self.best_acc:
                self.best_acc = valid_accuracy
                self.best_precision = valid_precision
                self.best_recall = valid_recall
                self.best_f1 = valid_f1

                print('Updated best accuracy and metrics.')

                print('The effect becomes better and the parameters are saved and start test...')
                # weight = r'result/model.pt'.format(model_chose)
                # torch.save(self.model.state_dict(), weight)
                print('\nValid set: Accuracy: ({:.5f}), Precision: ({:.5f}), Recall: ({:.5f}), F1: ({:.5f})'.format(
                    valid_accuracy, valid_precision, valid_recall, valid_f1))
                print(
                    'Best Accuracy: ({:.5f}), Best Precision: ({:.5f}), Best Recall: ({:.5f}), Best F1: ({:.5f})'.format(
                        self.best_acc, self.best_precision, self.best_recall, self.best_f1))




            return valid_accuracy

    def evaluate(self, epoch, test_loader, model_chose):
        import warnings
        warnings.filterwarnings("ignore")
        predicts_test, targets_test = [], []
        self.model.eval()
        with torch.no_grad():
            for test_idx, test_batch in enumerate(test_loader):
                test_batch = [x.to(self.device) for x in test_batch]
                ids_list, mask_list, predict_list, lstm_array, label_list = \
                    test_batch[0], test_batch[1], test_batch[2], test_batch[3], test_batch[4]

                if self.model_chose == 'Lstm':
                    out_scores = self.model.forward(x=lstm_array)
                    _, predicted = torch.max(out_scores, -1)

                if self.model_chose == 'LsTmCrf':
                    _, predicted = self.model.forward(rnn_array=lstm_array)

                if self.model_chose in ['Bert', 'BertLstM', 'AlBert', 'AlBertLstM', 'BertGRU', 'AlBertGRU']:
                    out_scores = self.model.forward(bert_ids=ids_list,
                                                    bert_mask=mask_list)
                    _, predicted = torch.max(out_scores, -1)

                if self.model_chose in ['BertCrf', 'BertLstMCrf', 'AlBertCrf', 'AlBertLstMCrf', 'BertGRUCrf', 'AlBertGRUCrf']:
                    _, predicted = self.model.forward(input_ids=ids_list, input_mask=mask_list)

                if self.model_chose in ['BertFusionAttCrf', 'AlBertFusionAttCrf']:
                    _, predicted = self.model.forward(input_ids=ids_list,
                                                      input_mask=mask_list,
                                                      lstm_array=lstm_array)

                predicted_list = torch.masked_select(predicted, predict_list)             #Torch. mask_select is used to remove all filled zeros, predicted is the original sequence, and predicted list is the predicted mask_list processed in batch_dataloader (indicating which ones are 1, which ones are 0, and 0 is removed)
                ture_ids = torch.masked_select(label_list, predict_list)
                predicts_test.extend(predicted_list.cpu().numpy().tolist())
                targets_test.extend(ture_ids.cpu().numpy().tolist())
            if model_chose == 'Lstm':
                targets_test, predicts_test = self.move_labels(predicts_test), self.move_labels(targets_test)

            accuracy = accuracy_score(targets_test, predicts_test)
            precision = precision_score(targets_test, predicts_test, average='macro')
            recall = recall_score(targets_test, predicts_test, average='macro')
            f1 = f1_score(targets_test, predicts_test, average='macro')


            os.makedirs('result', exist_ok=True)

            report = classification_report(y_true=targets_test, y_pred=predicts_test)
            result_text = r'result/test-result-{}.txt'.format(model_chose)
            file_handle = open(result_text, mode='a+', encoding='utf-8')
            file_handle.write('epoch:{},test_acc:{}, precision:{}, recall:{},f1_score:{}\n'.format(
                epoch, accuracy, precision, recall, f1
            ))
            file_handle.write('report:{}'.format(str(report)))
            file_handle.close()

    def move_labels(self, result_list):
        if 19 or 20 in result_list:
            k = []
            for x in result_list:
                if x == 19 or x == 20:
                    k.append(0)
                else:
                    k.append(x)
            return k
        else:
            return result_list

    def get_optimizer(self, model, lr):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        new_param = ['transitions', 'bert2hidden2label.weight', 'bert2hidden2label.bias', 'fusion_1_weight.weight',
                     'fusion_1_weight.bias', 'fc.weight', 'fc.bias']

        optimizer_grouped_parameters = [
            {'params':
                 [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not
                 any(nd in n for nd in new_param)],
             'weight_decay': 1e-5
             },
            {'params':
                 [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                  and not any(nd in n for nd in new_param)],
             'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        return optimizer


def main(model_chose, batch_size, epochs):
    print(f'choose model name >>>>>>------>>>>>>: {model_chose}')
    if torch.cuda.is_available():
        print(f'use gpu {torch.cuda.get_device_name()}')

    os.makedirs('reslut',exist_ok=True)

    train_set = BatchLoaderData(
        pickle_file='E:/paper_code/paper/project2/pickle_data/train_data.pickle')
    valid_set = BatchLoaderData(
        pickle_file='E:/paper_code/paper/project2/pickle_data/train_valid.pickle')
    test_set = BatchLoaderData(
        pickle_file='E:/paper_code/paper/project2/pickle_data/train_test.pickle')

    train_len = len(train_set)




    # 创建DataLoader实例
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    train2loss, valid2acc = [], []
    start = time.time()


    T = Trainer(model_chose, batch_size, epochs)

    for epoch in range(epochs):
        train_loss = T.train(epoch=epoch,
                             train_loader=train_loader,
                             len_train_set=train_len)

        valid_acc = T.valid(epoch=epoch,
                            valid_loader=valid_loader,
                            test_loader=test_loader,
                            model_chose=model_chose,
                            )

        if torch.cuda.is_available():
            train2loss.append(train_loss.cuda().data.cpu().numpy())
            valid2acc.append(valid_acc)
        else:
            train2loss.append(train_loss.detach().numpy())
            valid2acc.append(valid_acc)

            # 输出最佳指标
        print("\nBest Validation Metrics:")
        print(f"Accuracy: {T.best_acc:.5f}")
        print(f"Precision: {T.best_precision:.5f}")
        print(f"Recall: {T.best_recall:.5f}")
        print(f"F1 Score: {T.best_f1:.5f}")

        print("........................ Next ........................")

    end = time.time()
    result_text = r'result/test-result-{}.txt'.format(model_chose)
    file_handle = open(result_text, mode='a+', encoding='utf-8')
    file_handle.write(f'{model_chose} 训练时间长度为 {end - start} s\n')
    file_handle.close()

    train_loss_plt(train_loss=train2loss,
                   save_name=r"result/train-loss-{}".format(model_chose),
                   title=r"train-loss-{}".format(model_chose))
    valid_acc_plt(valid_acc=valid2acc,
                  save_name=r"result/valid-acc-{}".format(model_chose),
                  title=r"valid-acc-{}".format(model_chose))

    # 确保 'mat' 目录存在
    os.makedirs('mat', exist_ok=True)

    scio.savemat('mat/result_{}.mat'.format(model_chose), {'train_loss': train2loss, 'valid_acc': valid2acc})
if __name__ == '__main__':
    # 'Bert', 'BertCrf', 'BertLstM', 'BertLstMCrf', 'BertFusionAttCrf'              # todo bert
    # 'AlBert', 'AlBertCrf', 'AlBertLstM', 'AlBertLstMCrf', 'AlBertFusionAttCrf'    # todo albert
    # 'Lstm', 'LsTmCrf'                                                             # todo lstm
    # 'BertGRU','BertGRUCrf','AlBertGRU','AlBertGRUCrf'                             # todo GRU
    best_acc = 0

    main(model_chose='AlBertGRUCrf',  # todo 改模型名称，上面选择
         batch_size=64,
         epochs=30)


