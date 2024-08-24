from torch.utils.data import DataLoader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import argparse
import sys
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA

from metrics import get_metrics


def train(model, train_load, Learning_rate, LOSS_F, new_labels = None):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr= Learning_rate)

    trian_pbar = tqdm(enumerate(BackgroundGenerator(train_load)), total=len(train_load))

    train_losses_in_epoch = []

    if new_labels is None:
        for _, train_data in trian_pbar:
            trian_compounds, trian_proteins, trian_labels = train_data
            trian_compounds = trian_compounds.cuda()
            trian_proteins = trian_proteins.cuda()
            trian_labels = trian_labels.cuda()
            optimizer.zero_grad()

            predicts, scores = model.forward(trian_compounds, trian_proteins)
            # [256]，每个元素值代表样本预测为正样本的概率
            positive_pro = predicts[:, 1]
            train_loss = LOSS_F(positive_pro, trian_labels)
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
    else:
        for index, train_data in trian_pbar:
            trian_compounds, trian_proteins, _ = train_data
            trian_compounds = trian_compounds.cuda()
            trian_proteins = trian_proteins.cuda()
            trian_labels = new_labels[index].cuda()
            optimizer.zero_grad()

            # [256, 2]，每一行代表一个logit，即正负样本的预测概率
            predicts, scores = model.forward(trian_compounds, trian_proteins)
            train_loss = LOSS_F(predicts, trian_labels)
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

    train_loss_a_epoch = np.average(train_losses_in_epoch) 

    return model, train_loss_a_epoch

def test(model, test_load, LOSS_F):
    model.eval()

    test_pbar = tqdm(enumerate(BackgroundGenerator(test_load)), total=len(test_load))

    test_losses_in_epoch = []
    y_ture = torch.Tensor()
    y_pred = torch.Tensor()
    y_scores = torch.Tensor()
    with torch.no_grad():
        for _, test_data in test_pbar:
            test_compounds, test_proteins, test_labels = test_data
            test_compounds = test_compounds.cuda()
            test_proteins = test_proteins.cuda()

            predicts, scores = model.forward(test_compounds, test_proteins)
            #test_loss = LOSS_F(predicts.cpu(), test_labels)
            positive_pro = predicts[:, 1]
            test_loss = LOSS_F(positive_pro.cpu(), test_labels)
            test_losses_in_epoch.append(test_loss.item())

            y_ture = torch.cat((y_ture, test_labels), 0)
            y_pred = torch.cat((y_pred, predicts.cpu()), 0)
            y_scores = torch.cat((y_scores, scores.cpu()), 0)

    y_ture = y_ture.numpy()
    y_pred = np.argmax(y_pred.numpy(), axis=1)
    y_scores = y_scores.numpy()[:, 1]

    accuracy, precision, recall, f1, auc_roc, auc_pr = get_metrics(y_ture, y_pred, y_scores)
    test_loss_a_epoch = np.average(test_losses_in_epoch) 

    return test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr

def out_msg(epoch, epochs, test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr, step=None):
    epoch_len = len(str(epochs))
    if step:
        msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         step + 
                         f'test_loss: {test_loss_a_epoch:.5f} ' +
                         f'accuracy: {accuracy:.5f} ' +
                         f'precision: {precision:.5f} ' +
                         f'recall: {recall:.5f} ' +
                         f'f1: {f1:.5f} ' +
                         f'auc_roc: {auc_roc:.5f} ' +
                         f'auc_pr: {auc_pr:.5f} ')
    else:
        msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         f'test_loss: {test_loss_a_epoch:.5f} ' +
                         f'accuracy: {accuracy:.5f} ' +
                         f'precision: {precision:.5f} ' +
                         f'recall: {recall:.5f} ' +
                         f'f1: {f1:.5f} ' +
                         f'auc_roc: {auc_roc:.5f} ' +
                         f'auc_pr: {auc_pr:.5f} ')
    return msg

def write_to(loss_list = None, accuracy_list = None, title = None):
    with open('results.txt', 'a') as file:
        if title:
            file.write(title)
            file.write('\n')
        if loss_list:
            print(loss_list)
            file.write('loss：')
            for item in loss_list:
                file.write("{:.5f} ".format(item))
                #file.write(str(item) + ' ')
            file.write('   均值：{:.5f}'.format(sum(loss_list)/len(loss_list)))
            file.write('\n')
        if accuracy_list:
            print(accuracy_list)
            file.write('accuracy：')
            for item in accuracy_list:
                file.write("{:.5f} ".format(item))
                # file.write(str(item) + ' ')
            file.write('   均值：{:.5f}'.format(sum(accuracy_list)/len(accuracy_list)))
            file.write('\n')
            file.write('\n')

def parseArg():
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('-conf', metavar='conf_file', nargs=1, 
                        help='the config file for FedMD.'
                       )

    # 获取配置文件路径
    conf_file = os.path.abspath("conf.json")
    
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


# 采用均值为0，方差为5的高斯分布
def normal(x, mean=0, var=5):
    return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-((x - mean) ** 2) / (2 * var))

def proportion(K):
    values = np.array([normal(x) for x in np.linspace(-2, 2, K)])
    values = values / values.sum()
    return values





# 继承InMemoryDataset类，自定义一个将数据存储到内存的数据集类
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                xd=None, xt=None, y=None, transform=None,
                pre_transform=None,smile_graph=None):
        # root：用于存放原始、预处理数据
        # transform：数据转换函数，每一次数据获取时被调用
        # pre_transform：数据转换函数，数据保存到文件前被调用
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # 基准数据集
        self.dataset = dataset
        print(self.processed_paths[0])
        if os.path.isfile(self.processed_paths[0]):
            # 找到预处理数据
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            # 未找到预处理数据，固需处理数据
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回原始数据文件的文件名列表
    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    # 返回已处理数据文件的文件名列表
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    # 下载数据集原始文件到raw_dir文件夹
    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            # 将smiles转换为图
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            # 使用rdkit将SMILES转换为分子表示
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            # 为PyTorch Geometrics GCN算法做好图形准备
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            # 将化合物的3D图形、靶标序列和标签附加到数据列表
            data_list.append(GCNData)
        
        # 预过滤
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 预转换
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # 图形构建完成
        print('Graph construction done. Saving to file.')
        
        # 把数据划分成不同slices去保存读取 （大数据块切成小块），便于后续生成batch
        data, slices = self.collate(data_list)
        # save preprocessed data:
        # 保存预处理数据
        torch.save((data, slices), self.processed_paths[0])