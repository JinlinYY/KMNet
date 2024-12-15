import numpy as np
import torch
from sklearn.decomposition import PCA, MiniBatchSparsePCA, FastICA, IncrementalPCA, KernelPCA, MiniBatchDictionaryLearning
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from component.Cli_Encoder import extract_excel_features
from component.USI_Encoder import extract_image_features
from component.GNN_Encoder import gnn_extract_excel_features
from component.Fusion import combine_features
from metrics.plot_roc_curve import plot_roc_curve
from module.inputtotensor import inputtotensor
from component.Classifier import Classifier
from metrics.print_metrics import print_average_metrics, print_mean_std_metrics, print_average_metrics_BCa
from module.set_seed import set_seed
from module.train_test import train_test
from module.my_loss import FocalLoss
from metrics.plot_dac_curve import plot_dac_curve
import torch.nn as nn
from metrics.compute_specificity_fnr import compute_specificity_fnr, compute_specificity_fnr_SBR
from sklearn.metrics import  roc_curve, auc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    # 提取原始表格特征
    index, excel_feature, label = extract_excel_features('/tmp/pycharm_project_357/BCa/metadata-SBR.xlsx')
    # pca_excel = MiniBatchSparsePCA(n_components=15)
    # excel_feature_pca = pca_excel.fit_transform(excel_feature)
    excel_feature_pca_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # 提取超声图像特征
    image_filenames = ['/tmp/pycharm_project_357/BCa/CDIs_images_visualized/{}.png'.format(idx) for idx in index.astype(int)]
    image_features = extract_image_features(image_filenames, batch_size=16, num_workers=4)
    pca_image = MiniBatchSparsePCA(n_components=30)
    image_features_pca = pca_image.fit_transform(image_features)
    image_features_pca_tensor = torch.tensor(image_features_pca, dtype=torch.float32)


    # 表格特征构图，GNN提取图表格特征
    _, gnn_excel_feature, _ = gnn_extract_excel_features('/tmp/pycharm_project_357/BCa/metadata-SBR.xlsx')
    # pca_excel_gnn = MiniBatchSparsePCA(n_components=15)
    # gnn_excel_feature_pca = pca_excel_gnn.fit_transform(gnn_excel_feature)
    gnn_excel_feature_pca_tensor = torch.tensor(gnn_excel_feature, dtype=torch.float32)

    # 特征融合
    combined_features = combine_features(excel_feature_pca_tensor, image_features_pca_tensor, gnn_excel_feature_pca_tensor)  # 三模态
    # combined_features = combine_features(excel_feature_pca_tensor, image_features_pca_tensor)  # 两模态
    combined_features_tensor, label_tensor = inputtotensor(combined_features, label)



    # K-fold cross-validation
    k_folds = 4
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)


    # all_metrics = {"Validation": [], "Test": []}
    accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro, specificity_scores, FNR_scores = [], [], [], [], [], [], [], []
    all_y_true, all_y_probs = [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features, label):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor[train_index], combined_features_tensor[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]
        test_patient_indices = index[test_index]  # 获取测试集患者索引号
        net = Classifier(feature_dim=combined_features.shape[1], output_size=len(set(label))).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = FocalLoss(gamma=2.0, alpha=1.2)

        batch_size = 256
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

        # 计算并记录 specificity 和 FNR
        specificity, fnr = compute_specificity_fnr_SBR(y_test, test_probs, len(set(label)))
        specificity_scores.append(specificity)
        FNR_scores.append(fnr)

        # ROC curve and AUC for the current fold
        all_y_true.extend(y_test)
        all_y_probs.extend(test_probs)
        roc_auc_fold = plot_roc_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test", save_path="ROC_fig/roc_curve.png")
        AUC_score_macro.append(roc_auc_fold['macro'])
        AUC_score_micro.append(roc_auc_fold['micro'])
        # 绘制DAC曲线
        dac_metrics = plot_dac_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test",
                                     save_path=f"DAC_fig/dac_curve_fold{fold}.png")
        # print(f"DAC metrics for fold {fold}: {dac_metrics}")


    #  ROC  AUC
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    overall_roc_auc = plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall", save_path="ROC_fig/roc_curve.png")
    # print(overall_roc_auc)
    # 绘制总体DAC曲线
    overall_dac_metrics = plot_dac_curve(all_y_true, all_y_probs, dataset_type="Overall", save_path="DAC_fig/dac_curve.png")
    # print(f"Overall DAC metrics: {overall_dac_metrics}")
    # 打印平均指标
    print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro, specificity_scores, FNR_scores)


if __name__ == "__main__":
    SEED = 45
    set_seed(SEED)
    main()
