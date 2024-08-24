import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

"""
    1、客户端模型初始化
    2、客户端模型预测
    3、聚合预测
    4、服务器模型更新
    5、客户端模型更新
"""

from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import torch

from models.DeepDTA import DeepDTA, remove_last_layer
from preprocess_data import collate_fn
from early_stop import EarlyStopping
from utils import train, test, out_msg, write_to, parseArg, init_weights
from generate_data import generate_collaborative_data, shuffle_dataset, Non_IID_protein, random_divide


from dataset.kiba import get_DTIs, get_proteins, get_drugs

if __name__ == "__main__":
    # 读取配置文件
    conf_file =  parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        K = conf_dict["K"]

        Centralized = conf_dict["Centralized"]
        pre_epochs = conf_dict["pre_epochs"]
        pre_lr = conf_dict["pre_lr"]
        pre_batch_size = conf_dict["pre_batch_size"]
        pre_patience = conf_dict["pre_patience"]
        pre_delta = conf_dict["pre_delta"]

        Local = conf_dict["Local"]
        init_epochs = conf_dict["init_epochs"]
        init_lr = conf_dict["init_lr"]
        init_batch_size = conf_dict["init_batch_size"]
        init_patience = conf_dict["init_patience"]
        init_delta = conf_dict["init_delta"]

        Ours = conf_dict["FedKD-DTI"]
        collaborate_Epoch = conf_dict["collaborate_Epoch"]
        N_collaborative = conf_dict["N_collaborative"]
        pri_patience = conf_dict["pri_patience"]
        pri_delta = conf_dict["pri_delta"]

        alignment_epoch = conf_dict["alignment_epoch"]
        alignment_batch_size = conf_dict["alignment_batch_size"]
        alignment_lr = conf_dict["alignment_lr"]

        private_training_epoch = conf_dict["private_training_epoch"]
        private_training_batch_size = conf_dict["private_training_batch_size"]
        private_training_lr = conf_dict["private_training_lr"]
    del conf_dict, conf_file

    # 输出分类、logits的客户端
    classifier_parties = []
    logits_parties = []
    for i in range(K):
        model = DeepDTA().cuda()
        classifier_parties.append(model)
        model = remove_last_layer(model)
        logits_parties.append(model)

    # 输出分类、logits的服务端
    server_classify = DeepDTA().cuda()
    server_logit = remove_last_layer(server_classify)

    # 客户端、服务端模型参数字典
    client_model_state_dict = [0] * K
    server_model_state_dict = [0] * 1

    drugs_dict = get_drugs()
    proteins_dict = get_proteins()
    DTIs = get_DTIs(drugs_dict, proteins_dict)

    # 随机划分，两个测试集
    #public_train, public_test, private_train_list, private_test = generate(DTIs, K)
    #public_train, public_test, private_train_list, private_test_list = random_divide(DTIs, K)
    public_train, public_test, private_train_list, private_test_list = random_divide(DTIs, K)

    LOSS_F1 = nn.BCELoss()                  # 二分类交叉熵
    LOSS_F2 = nn.L1Loss()                   # 平均绝对误差
    LOSS_F3 = nn.MSELoss()                  # 均方误差
    

    """
    Centralized Learning
    """
    if Centralized:
        print("start centralized learning: ")
        train_data = []
        train_data.extend(public_train)
        test_data = []
        test_data.extend(public_test)

        # 获取中心式训练集、测试集
        for i in range(K):
            train_data.extend(private_train_list[i])
            #test_data.extend(private_test_list[i])
            
        train_load = DataLoader(train_data, batch_size=pre_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        test_load = DataLoader(test_data, batch_size=pre_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

        model = classifier_parties[0]
        early_stopping = EarlyStopping(pre_patience, pre_delta, verbose= True)

        for epoch in range(1, pre_epochs+1):
            model, train_loss_a_epoch = train(model, train_load, pre_lr, LOSS_F1)
            test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr = test(model, test_load, LOSS_F1)
            msg = out_msg(epoch, pre_epochs, test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr, step=None)
            print(msg)

            early_stopping(test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr, model)
            if early_stopping.early_stop or epoch == pre_epochs:
                print("stopping!!!")
                # 输出平均aupr值
                print("loss: {}".format(early_stopping.accompany_loss)) 
                print("accuracy: {}".format(early_stopping.accompany_accuracy)) 
                print("precision: {}".format(early_stopping.accompany_precision)) 
                print("recall: {}".format(early_stopping.accompany_recall)) 
                print("f1: {}".format(early_stopping.accompany_f1)) 
                print("auc: {}".format(early_stopping.accompany_auc_roc)) 
                print("aupr: {}".format(early_stopping.best_auc_pr)) 
                break


    """
    Local Learning
    """
    if Local:
        print("start clients model initialization: ")
        init_loss_list = [0 for i in range(K)]
        init_accuracy_list = [0 for i in range(K)]
        init_precision_list = [0 for i in range(K)]
        init_recall_list = [0 for i in range(K)]
        init_f1_list = [0 for i in range(K)]
        init_auc_roc_list = [0 for i in range(K)]
        init_auc_pr_list = [0 for i in range(K)]

        for i in range(K):
            print('Client {} is training'.format(i+1))
            model = classifier_parties[i]
            early_stopping = EarlyStopping(init_patience, init_delta, verbose= True)

            # 训练、测试集
            private_train = private_train_list[i]
            private_train = shuffle_dataset(private_train, 1234)
            private_test = private_test_list[i]

            train_load = DataLoader(private_train, batch_size=init_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
            test_load = DataLoader(private_test, batch_size=init_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
            for epoch in range(1, init_epochs+1):
                # train、test、msg
                model, train_loss_a_epoch = train(model, train_load, init_lr, LOSS_F1)
                test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr = test(model, test_load, LOSS_F1)
                msg = out_msg(epoch, init_epochs, test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr, step=None)
                print(msg)

                # stop
                early_stopping(test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr, model)
                if early_stopping.early_stop or epoch == init_epochs:
                    print("stopping!!!")
                    init_loss_list[i] = early_stopping.accompany_loss
                    init_accuracy_list[i] = early_stopping.accompany_accuracy
                    init_precision_list[i] = early_stopping.accompany_precision
                    init_recall_list[i] = early_stopping.accompany_recall
                    init_f1_list[i] = early_stopping.accompany_f1
                    init_auc_roc_list[i] = early_stopping.accompany_auc_roc
                    init_auc_pr_list[i] = early_stopping.best_auc_pr
                    client_model_state_dict[i] = model.state_dict()
                    break

        # 输出平均aupr值
        print("aupr list: {}".format(init_auc_pr_list))
        print("mean loss: {}".format(sum(init_loss_list)/len(init_loss_list))) 
        print("mean accuracy: {}".format(sum(init_accuracy_list)/len(init_accuracy_list))) 
        print("mean precision: {}".format(sum(init_precision_list)/len(init_precision_list))) 
        print("mean recall: {}".format(sum(init_recall_list)/len(init_recall_list))) 
        print("mean f1: {}".format(sum(init_f1_list)/len(init_f1_list))) 
        print("mean auc: {}".format(sum(init_auc_roc_list)/len(init_auc_roc_list))) 
        print("mean aupr: {}".format(sum(init_auc_pr_list)/len(init_auc_pr_list))) 


    """
    FedKD-DTI
    """
    if Ours:

        if not Local:
            # 加载模型参数
            for i in range(K):
                #classifier_parties[i].load_state_dict(torch.load('./models_parameters/initialization_10/client_{}.pth'.format(i)))
                #client_model_state_dict[i] = classifier_parties[i].state_dict()
                model = classifier_parties[i]
                model.apply(init_weights)
                client_model_state_dict[i] = model.state_dict()
            # 随机初始化服务端模型参数
            server_classify.apply(init_weights)
            server_model_state_dict[0] = server_classify.state_dict()
            

        """协作数据集：这里采用的是整个公共数据集作为协作数据集"""
        print("...协作数据集...")
        col_data = generate_collaborative_data(public_train, len(public_train))
        col_dataset_load = DataLoader(col_data, batch_size = 256, shuffle=False, num_workers=0, collate_fn=collate_fn)

        server_preds = []

        early_stopping = EarlyStopping(pri_patience, pri_delta, verbose= True)

        for epoch in range(1, collaborate_Epoch+1):

            print("...第{}次迭代...".format(epoch))

            """客户端蒸馏"""
            if epoch != 1:
                print("...客户端蒸馏...")
                # 除第一迭代之外的所有次迭代，客户端都需在服务端发送的预测值进行蒸馏
                for i in range(K):
                    print('client {} is training...'.format(i+1))
                    model = logits_parties[i]
                    model.load_state_dict(client_model_state_dict[i])
                    for local_epoch in range(alignment_epoch):
                        model, train_loss_a_epoch = train(model, col_dataset_load, alignment_lr, LOSS_F2, server_preds)
                    client_model_state_dict[i] = model.state_dict()


            """客户端训练"""
            print("...客户端训练...")
            for i in range(K):
                print('client {} is training...'.format(i+1))
                model = classifier_parties[i]
                model.load_state_dict(client_model_state_dict[i])
                #private_train = private_train_list[i]
                private_train = private_train_list[i] + public_train
                private_train = shuffle_dataset(private_train, 1234)
                client_train_load = DataLoader(private_train, batch_size=private_training_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
                for local_epoch in range(1, private_training_epoch+1):
                    model, train_loss_a_epoch = train(model, client_train_load, private_training_lr, LOSS_F1)
                client_model_state_dict[i] = model.state_dict()

            
            """客户端测试"""
            print("...客户端测试...")
            private_loss_list = [0 for i in range(K)]
            private_accuracy_list = [0 for i in range(K)]
            private_precision_list = [0 for i in range(K)]
            private_recall_list = [0 for i in range(K)]
            private_f1_list = [0 for i in range(K)]
            private_auc_list = [0 for i in range(K)]
            private_aupr_list = [0 for i in range(K)]
            for i in range(K):
                model = classifier_parties[i]
                model.load_state_dict(client_model_state_dict[i])

                # 每个客户端的测试集不同
                private_test = private_test_list[i]

                test_load = DataLoader(private_test, batch_size=private_training_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
                test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr = test(model, test_load, LOSS_F1)
                msg = out_msg(epoch, init_epochs, test_loss_a_epoch, accuracy, precision, recall, f1, auc_roc, auc_pr, step=None)
                print(msg)

                private_loss_list[i] = test_loss_a_epoch
                private_accuracy_list[i] = accuracy
                private_precision_list[i] = precision
                private_recall_list[i] = recall
                private_f1_list[i] = f1
                private_auc_list[i] = auc_roc
                private_aupr_list[i] = auc_pr

            mean_loss = sum(private_loss_list)/K
            mean_accuracy = sum(private_accuracy_list)/K
            mean_precision = sum(private_precision_list)/K
            mean_recall = sum(private_recall_list)/K
            mean_f1 = sum(private_f1_list)/K
            mean_auc = sum(private_auc_list)/K
            mean_aupr = sum(private_aupr_list)/K

            early_stopping(mean_loss, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc, mean_aupr, model)
            if early_stopping.early_stop or epoch == collaborate_Epoch:
                print("stopping!!!")
                print("mean loss: {}".format(early_stopping.accompany_loss)) 
                print("mean accuracy: {}".format(early_stopping.accompany_accuracy)) 
                print("mean precision: {}".format(early_stopping.accompany_precision)) 
                print("mean recall: {}".format(early_stopping.accompany_recall)) 
                print("mean f1: {}".format(early_stopping.accompany_f1)) 
                print("mean auc: {}".format(early_stopping.accompany_auc_roc)) 
                print("mean aupr: {}".format(early_stopping.best_auc_pr))
                break


            """客户端预测"""
            print("...客户端预测...")
            clients_preds = []
            for i in range(K):
                print('client {} is predicting'.format(i+1))
                model = logits_parties[i]
                model.load_state_dict(client_model_state_dict[i])
                model.eval()

                col_pbar = tqdm(enumerate(BackgroundGenerator(col_dataset_load)), total=len(col_dataset_load))

                single_preds = []   # 长度为批量个数
                with torch.no_grad():
                    for index, col_data_item in col_pbar:
                        col_compounds, col_proteins, col_labels = col_data_item
                        col_compounds = col_compounds.cuda()
                        col_proteins = col_proteins.cuda()
                        col_predictions, scores = model.forward(col_compounds, col_proteins)
                        single_preds.append(col_predictions.cpu())
                        del col_compounds, col_proteins, col_predictions
                clients_preds.append(single_preds)

            total_sum = sum(len(sublist) for sublist in private_train_list)
            """聚合预测值"""
            print("...聚合预测值...")
            count = len(clients_preds[0])       # 批量数
            mean_preds = [0] * count        # 批量进行平均
            for i in range(count):          # 遍历每个批量
                empty_tensor = torch.zeros(clients_preds[0][i].size())      # 与预测值张量形状一致
                for j in range(K):                      # 求和
                    w = len(private_train_list[j])/total_sum
                    empty_tensor = empty_tensor + w*clients_preds[j][i]     # 加权平均
                    #empty_tensor = empty_tensor + clients_preds[j][i]
                mean_preds[i] = empty_tensor
                #mean_preds[i] = empty_tensor / K        # 平均


            """服务端蒸馏"""
            print("...服务端蒸馏...")
            server_logit.load_state_dict(server_model_state_dict[0])
            for local_epoch in range(alignment_epoch):
                server_logit, train_loss_a_epoch = train(server_logit, col_dataset_load, alignment_lr, LOSS_F2, mean_preds)
            server_model_state_dict[0] = server_logit.state_dict()


            """服务端训练"""
            print("...服务端训练...")
            server_classify.load_state_dict(server_model_state_dict[0])
            server_train_load = DataLoader(public_train, batch_size=private_training_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
            for local_epoch in range(private_training_epoch):
                server_classify, train_loss_a_epoch = train(server_classify, server_train_load, private_training_lr, LOSS_F1)
            server_model_state_dict[0] = server_classify.state_dict()


            """服务端预测"""
            print("...服务端预测...")
            server_logit.load_state_dict(server_model_state_dict[0])
            server_logit.eval()

            col_pbar = tqdm(enumerate(BackgroundGenerator(col_dataset_load)), total=len(col_dataset_load))

            server_preds = []
            with torch.no_grad():
                for index, col_data_item in col_pbar:
                    col_compounds, col_proteins, col_labels = col_data_item
                    col_compounds = col_compounds.cuda()
                    col_proteins = col_proteins.cuda()
                    col_predictions, scores = server_logit.forward(col_compounds, col_proteins)
                    server_preds.append(col_predictions.cpu())