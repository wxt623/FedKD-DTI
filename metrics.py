from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


# 返回Accuracy, Precision, Reacll, F1, AUC, PRC
def get_metrics(y_true, y_pred, y_scores):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确率
    precision = precision_score(y_true, y_pred, zero_division=1)

    # 计算召回率
    recall = recall_score(y_true, y_pred)

    # 计算F1分数
    f1 = f1_score(y_true, y_pred)

    # 计算AUC
    auc_roc = roc_auc_score(y_true, y_scores)

    # 计算AUPR
    auc_pr = average_precision_score(y_true, y_scores)

    return accuracy, precision, recall, f1, auc_roc, auc_pr
