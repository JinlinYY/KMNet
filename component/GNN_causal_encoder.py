import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义12个节点
node_names = ['Age', '乳房手术', '腋窝手术', '病理学类型', 'size（cm）', 'diff', '乳头或皮肤受累',
              'LVI', 'ER', 'PR', 'Ki-67%', 'EGFR']

# 定义先验边（有向）
edges_prior = [
    ('病理学类型', 'Ki-67%'),
    ('size（cm）', 'Ki-67%'),
    ('diff', 'Ki-67%'),
    ('Ki-67%', 'LVI'),
    ('LVI', '乳房手术'),
    ('LVI', '腋窝手术'),
    ('乳头或皮肤受累', '乳房手术'),
    ('乳头或皮肤受累', '腋窝手术'),
    ('size（cm）', '乳房手术'),
    ('size（cm）', '腋窝手术'),
]

# 建立节点名称到索引的映射
name2idx = {name: idx for idx, name in enumerate(node_names)}

# 构建边索引张量
edge_index_base = torch.tensor(
    [[name2idx[src], name2idx[tgt]] for src, tgt in edges_prior],
    dtype=torch.long
).t().contiguous()

def add_self_loops(edge_index, num_nodes):
    loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index

def gnn_extract_excel_features(filename):
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')
    index = readbook.iloc[:, 0].to_numpy()
    labels = readbook.iloc[:, -1].to_numpy()
    features_df = readbook.iloc[:, 1:-1]
    # 打印列名和特征数量
    print(f"Feature columns: {features_df.columns}")
    print(f"Number of features: {features_df.shape[1]}")
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])
    # 输出数值特征和类别特征的数量
    print(f"Number of numeric features: {numeric_features.shape[1]}")
    print(f"Number of categorical features: {categorical_features.shape[1]}")

    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
        print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")

    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)
    # 打印最终合并特征的数量
    print(
        f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")

    def create_graph_from_features(features):
        num_patients = features.shape[0]
        num_features_per_node = features.shape[1]

        edge_index_list = []
        node_features_list = []
        batch_list = []

        for patient_idx in range(num_patients):
            node_features = torch.tensor(features[patient_idx], dtype=torch.float32).repeat(len(node_names), 1)
            node_features_list.append(node_features)

            edge_index = edge_index_base.clone()
            edge_index = add_self_loops(edge_index, len(node_names))
            edge_index_list.append(edge_index)

            batch_list.extend([patient_idx] * len(node_names))

        all_node_features = torch.cat(node_features_list, dim=0)
        all_edge_indices = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long)
        print(f"batch_tensor {batch_tensor}")
        print(f"num_patients {num_patients}")
        # 确保 batch_tensor 的索引没有超出范围
        max_batch_index = batch_tensor.max().item()
        if max_batch_index >= num_patients:
            raise ValueError(f"Batch index {max_batch_index} exceeds the number of patients ({num_patients})")

        # 使用 Batch 来将图数据合并，保持不同患者的独立性
        graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)
        return graph_data

    graph_data = create_graph_from_features(combined_features)
    # 检查 x 的维度
    print(f"x shape: {graph_data.x.shape}")
    print(f"edge_index shape: {graph_data.edge_index.shape}")
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.conv2 = GCNConv(128, 64)
            self.conv3 = GCNConv(64, out_channels)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)  # ← 每个图做平均池化，得到 [num_patients, feature_dim]
            return x

    model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
    data = graph_data.to(device)

    model.eval()
    with torch.no_grad():
        aggregated_features = model(data.x, data.edge_index, data.batch).cpu().numpy()

    return index, aggregated_features, labels
