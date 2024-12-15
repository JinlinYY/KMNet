import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
def plot_roc_curve(y_true, y_probs, dataset_type="Test", save_path=None):
    """
    绘制ROC曲线并打印AUC值
    """
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    # 定义类别名称
    # class_labels = ["LN0", "LN1-3", "LN4+"]
    class_labels = ["HER2-low", "HER2-zero", "HER2-positive"]
    # 将y_true转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均 ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    # 计算微平均 ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 打印AUC值
    print(f"{dataset_type} ROC AUC values:")
    for i in range(n_classes):
        print(f"Class {class_labels[i]} AUC: {roc_auc[i]:.2f}")
    print(f"Macro Average AUC: {roc_auc['macro']:.2f}")
    print(f"Micro Average AUC: {roc_auc['micro']:.2f}")

    # 绘制ROC曲线
    plt.figure(figsize=(9, 7))

    colors = ['#0072C6', '#C60000', '#FFA500', '#d62728', '#9467bd']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_labels[i]} ROC curve (AUC = {roc_auc[i]:.2f})')


    # plt.plot(fpr["macro"], tpr["macro"], color='#000080', linestyle='-.', lw=4, label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], color='#9467bd', linestyle='-.', lw=4,
             label=f'Average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    # plt.plot(fpr["micro"], tpr["micro"], color='#800080', linestyle='--', lw=4, label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_type} ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # 如果提供了保存路径，则保存图片
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png')
        print(f"ROC曲线已保存到: {save_path}")
    else:
        plt.show()

    # 返回计算的 AUC 值字典
    return roc_auc


def plot_roc_curve_2(all_y_true, all_y_probs, dataset_type="Overall", save_path="ROC_fig/roc_curve.png"):
    """
    绘制并保存 ROC 曲线。

    参数:
        all_y_true (list or numpy array): 真实标签（0 或 1）。
        all_y_probs (list or numpy array): 模型预测概率。
        dataset_type (str): 数据集类型，用于图标题（默认 "Overall"）。
        save_path (str): 保存图像的路径（默认 "DAC_fig/dac_curve.png"）。
    """
    # 计算 ROC 曲线和 AUC 值
    fpr, tpr, _ = roc_curve(all_y_true, all_y_probs)
    roc_auc = auc(fpr, tpr)

    # 确保保存路径的目录存在
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_type}")
    plt.legend(loc="lower right")

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_path}")

    return roc_auc

