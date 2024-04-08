import numpy as np
import random
from collections import Counter
import os
import csv
from utils import proportion

SEED = 1234

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


""" 随机抽取协作数据集 """
def generate_collaborative_data(public_data, N):
    random.seed(SEED)
    return random.sample(public_data, N)


"""生成客户端数据文件"""
def generate_client_csv(train_data_list, test_data_list, K, path):

    header = ["compound_iso_smiles", "target_sequence", "affinity"]

    for i in range(K):
        client_path = path + '/client_' + str(i)
        os.makedirs(client_path, exist_ok=True)

        train_data = train_data_list[i]
        train_path = client_path + '/kiba_train.csv'
        with open(train_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for pair, label in train_data:
                writer.writerow([pair[0], pair[1], label])

        test_data = test_data_list[i]
        test_path = client_path + '/kiba_test.csv'
        with open(test_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for pair, label in test_data:
                writer.writerow([pair[0], pair[1], label])


"""生成服务端数据文件"""
def generate_server_csv(train_data, test_data, K, path):
    header = ["compound_iso_smiles", "target_sequence", "affinity"]
    
    server_path = path + '/server'
    os.makedirs(server_path, exist_ok=True)

    train_path = server_path + '/kiba_train.csv'
    with open(train_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for pair, label in train_data:
                writer.writerow([pair[0], pair[1], label])

    test_path = server_path + '/kiba_test.csv'
    with open(test_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pair, label in test_data:
            writer.writerow([pair[0], pair[1], label])


"""读取文件"""
def read(dataset, k, type):
    data = []

    filename = "/homec/wangxuetao/data/DTI-2.0/{}/client_{}/kiba_{}.csv".format(dataset, k, type)
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        is_header = True
        for row in reader:
            # 去掉表头
            if is_header:
                is_header = False
                continue
            data.append(row)

    return data


"""通过DTIs，获取以蛋白质为中心的数据字典"""
def get_DTI_dict(DTIs):
    DTIs_dcit_by_protein = {}

    for item in DTIs:
        pair, label = item
        drug, protein = pair

        """# 以蛋白质为key，药物-靶标对作为value
        if protein not in DTIs_dcit_by_protein:
            DTIs_dcit_by_protein[protein] = []
            DTIs_dcit_by_protein[protein].append([[drug, protein], label])
        else:
            DTIs_dcit_by_protein[protein].append([[drug, protein], label])"""
        
        # 以药物为key，药物-靶标对作为value
        if drug not in DTIs_dcit_by_protein:
            DTIs_dcit_by_protein[drug] = []
            DTIs_dcit_by_protein[drug].append([[drug, protein], label])
        else:
            DTIs_dcit_by_protein[drug].append([[drug, protein], label])

    return DTIs_dcit_by_protein


"""Non-IID分布"""
def Non_IID_protein(dataset, K):
    dataset = shuffle_dataset(dataset, SEED)

    public_train= dataset[0 : int(len(dataset)*0.23)]                        # 公共训练集

    public_test = dataset[int(len(dataset)*0.23) : int(len(dataset)*0.3)]    # 共用测试集

    total_private_train = dataset[int(len(dataset)*0.3) :]                   # 私有训练集总和

    private_train_list = [None]*K
    private_test_list = [None]*K

    DTIs_dcit_by_protein = get_DTI_dict(total_private_train)

    keys = list(DTIs_dcit_by_protein.keys())                    # 获取所有蛋白质
    keys = shuffle_dataset(keys, 10)
    length = len(keys)

    values = proportion(K)                                      # 各个客户端所占蛋白质的比重
    values.sort()
    counts = [round(length * i) for i in values]

    #print("Current protein distribution: {}".format(counts))
    print("Current drug distribution: {}".format(counts))

    for i in range(K):

        private_train = []

        if i == K-1:    # 第K个客户端
            temp_keys = keys
        else:           # 第1..K-1个客户端
            count = counts[i]
            temp_keys = keys[:count]
            del keys[:count]

        for key in temp_keys:                                    # 客户端本地含有的蛋白质
            data = DTIs_dcit_by_protein[key]
            private_train = private_train + data
        
        private_train = shuffle_dataset(private_train, SEED)
        private_train_list[i] = private_train                     # 私有测试集
        private_test_list[i] = public_test                       # 私有训练集

    return public_train, public_test, private_train_list, private_test_list


"""IID分布"""
def random_divide(dataset, K):
    dataset = shuffle_dataset(dataset, SEED)

    public_train= dataset[0 : int(len(dataset)*0.25)]                        # 公共数据集

    public_test = dataset[int(len(dataset)*0.25) : int(len(dataset)*0.3)]    # 共用测试集

    total_private_train = dataset[int(len(dataset)*0.3) :]                   # 私有数据集总和

    private_train_list = []
    private_test_list = []

    length_per_private_train = len(total_private_train) // K
    for i in range(K):
        
        private_train = total_private_train[i*length_per_private_train: (i+1)*length_per_private_train]     # 客户端k私有数据集

        private_train_list.append(private_train)
        private_test_list.append(public_test)

    return public_train, public_test, private_train_list, private_test_list

