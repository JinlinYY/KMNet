import numpy as np
import torch
from sklearn.decomposition import PCA, MiniBatchSparsePCA, FastICA, IncrementalPCA, KernelPCA, MiniBatchDictionaryLearning
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from component.Cli_Encoder import extract_excel_features
from component.USI_Encoder import extract_image_features
from component.GNN_Encoder import gnn_extract_excel_features
from component.Fusion import combine_features
from metrics.plot_roc_curve import plot_roc_curve, plot_roc_curve_2
from module.inputtotensor import inputtotensor
from component.Classifier import Classifier
from metrics.print_metrics import print_average_metrics, print_mean_std_metrics, print_average_metrics_BCa
from module.set_seed import set_seed
from module.train_test import train_test
from module.my_loss import FocalLoss
from metrics.plot_dac_curve import plot_dac_curve, plot_dac_curve_2
import torch.nn as nn
from metrics.compute_specificity_fnr import compute_specificity_fnr, compute_specificity_fnr_2
from sklearn.metrics import  roc_curve, auc
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_features(features, filepath):
    """保存特征到文件"""
    np.save(filepath, features)


def load_features(filepath):
    """加载特征文件"""
    if os.path.exists(filepath):
        return np.load(filepath)
    return None

def main():
    train_excel_path = '/tmp/pycharm_project_357/ISIC/train-del.xlsx'


    # 保存特征的文件路径
    excel_feature_train_file = './ISIC/features/excel_feature_train.npy'
    image_features_train_file = './ISIC/features/image_features_train.npy'
    gnn_excel_feature_train_file = './ISIC/features/gnn_excel_feature_train.npy'

    combined_features_train_file = './ISIC/features/combined_features_train.npy'


    # 加载训练集原始表格特征
    index_train, excel_feature_train, label_train = extract_excel_features(train_excel_path)
    # 处理训练集特征
    if load_features(excel_feature_train_file) is None:
        # PCA降维
        pca_excel = MiniBatchSparsePCA(n_components=50)
        excel_feature_pca_train = pca_excel.fit_transform(excel_feature_train)

        save_features(excel_feature_pca_train, excel_feature_train_file)  # 保存特征
        print("save_features(excel_feature_pca_train, excel_feature_train_file) is ok!")
    else:
        excel_feature_pca_train = load_features(excel_feature_train_file)
        print("load_features(excel_feature_train_file) is ok!")

    # 提取训练集超声图像特征
    if load_features(image_features_train_file) is None:
        image_filenames_train = ['/tmp/pycharm_project_357/ISIC/train_128/{}.jpg'.format(idx) for idx in
                                 index_train.astype(int)]
        image_features_train = extract_image_features(image_filenames_train, batch_size=16, num_workers=4)
        pca_image = MiniBatchSparsePCA(n_components=50)
        image_features_pca_train = pca_image.fit_transform(image_features_train)

        save_features(image_features_pca_train, image_features_train_file)  # 保存图像特征
        print("save_features(image_features_pca_train, image_features_train_file) is ok!")
    else:
        image_features_pca_train = load_features(image_features_train_file)
        print("load_features(image_features_train_file) is ok!")

    # 提取训练集表格特征（GNN）
    if load_features(gnn_excel_feature_train_file) is None:
        _, gnn_excel_feature_train, _ = gnn_extract_excel_features(train_excel_path)
        pca_gnn = MiniBatchSparsePCA(n_components=50)
        gnn_excel_feature_pca_train = pca_gnn.fit_transform(gnn_excel_feature_train)

        save_features(gnn_excel_feature_pca_train, gnn_excel_feature_train_file)  # 保存GNN特征
        print("save_features(gnn_excel_feature_pca_train, gnn_excel_feature_train_file) is ok!")
    else:
        gnn_excel_feature_pca_train = load_features(gnn_excel_feature_train_file)
        print("load_features(gnn_excel_feature_train_file) is ok!")


    # 特征融合
    if load_features(combined_features_train_file) is None:
        combined_features_train = combine_features(excel_feature_pca_train, image_features_pca_train,
                                                   gnn_excel_feature_pca_train)
        save_features(combined_features_train, combined_features_train_file)  # 保存融合后的训练特征
        print("save_features(combined_features_train, combined_features_train_file) is ok!")
    else:
        combined_features_train = load_features(combined_features_train_file)
        print("load_features(combined_features_train_file) is ok!")



    # 将特征转换为tensor
    combined_features_tensor_train, label_tensor = inputtotensor(combined_features_train, label_train)



    # K-fold cross-validation
    k_folds = 20
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores, specificity_scores, FNR_scores = [], [], [], [], [], [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features_train, label_train):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor_train[train_index], combined_features_tensor_train[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]
        print(f"combined_features.shape[1]:{combined_features_train.shape[1]}")

        test_patient_indices = index_train[test_index]  # 获取测试集患者索引号


        net = Classifier(feature_dim=combined_features_train.shape[1], output_size=len(set(label_train))).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = FocalLoss(gamma=2.0, alpha=1.2)

        batch_size = 2048
        model_path = f'./pth/best_model_fold{fold}.pth'

        cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred, train_losses, val_losses = train_test(
            x_train, y_train, x_test, y_test,
            x_test, y_test,
            net, optimizer, loss_func, batch_size, model_path
        )

        # 打印测试集预测结果
        print(f"Test patient indices (fold {fold}): {test_patient_indices}")
        print(f"Fold {fold} - Test set true labels: {y_test.numpy()}")
        print(f"Fold {fold} - Test set predicted labels: {y_test_pred}")
        print(f"Fold {fold} - Test set predicted probabilities: {test_probs}")

        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_test_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_test_pred, average='weighted'))
        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)
        specificity, fnr = compute_specificity_fnr_2(y_test, y_test_pred)
        specificity_scores.append(specificity)
        FNR_scores.append(fnr)

    # 打印平均指标
    print_average_metrics_BCa(accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores, specificity_scores, FNR_scores)

if __name__ == "__main__":
    SEED = 15
    set_seed(SEED)
    main()
