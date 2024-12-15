# ##########节点全连接################
#
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.nn import global_mean_pool
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def add_self_loops(edge_index, num_nodes):
#     """
#     添加自环边（每个节点都和自己有一条边连接）
#     """
#     loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
#     edge_index = torch.cat([edge_index, loop_index], dim=1)
#     return edge_index
#
#
# def gnn_extract_excel_features(filename):
#     # 读取Excel数据
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#
#     # 患者 ID 和标签
#     index = readbook.iloc[:, 0].to_numpy()  # 患者ID
#     labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）
#
#     # 临床病理特征（去掉患者ID和标签）
#     features_df = readbook.iloc[:, 1:-1]
#
#     # 打印列名和特征数量
#     print(f"Feature columns: {features_df.columns}")
#     print(f"Number of features: {features_df.shape[1]}")
#
#     # 分析数值特征和类别特征
#     numeric_features = features_df.select_dtypes(include=[np.number])
#     categorical_features = features_df.select_dtypes(exclude=[np.number])
#
#     # 输出数值特征和类别特征的数量
#     print(f"Number of numeric features: {numeric_features.shape[1]}")
#     print(f"Number of categorical features: {categorical_features.shape[1]}")
#
#     # 对类别特征进行独热编码
#     if not categorical_features.empty:
#         categorical_features = pd.get_dummies(categorical_features)
#         print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")
#
#     # 合并数值特征和类别特征
#     combined_features = pd.concat([numeric_features, categorical_features], axis=1)
#     combined_features = combined_features.to_numpy(dtype=np.float32)
#     # 打印最终合并特征的数量
#     print(
#         f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")
#
#     # 构建图数据：每个患者为一个图，每个临床特征为一个节点
#     def create_graph_from_features(features):
#         num_patients = features.shape[0]  # 患者数量
#         num_features = features.shape[1]  # 每个患者的临床特征数量（每列一个特征）
#
#         edge_index_list = []  # 存储所有患者的边信息
#         node_features_list = []  # 存储所有患者的节点特征
#         batch_list = []  # 存储患者的批处理信息
#
#         for patient_idx in range(num_patients):
#             patient_features = features[patient_idx]  # 当前患者的特征
#             node_features_list.append(patient_features)
#
#             # 构建边：基于特征间的关系
#             edge_index = []
#             for i in range(num_features):
#                 for j in range(num_features):
#                     if i != j:  # 不创建自环，稍后会统一添加
#                         edge_index.append([i, j])
#
#             # 转换为 PyTorch 张量并添加自环
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             edge_index = add_self_loops(edge_index, num_features)
#             edge_index_list.append(edge_index)
#
#             # 为每个患者设置一个独立的批处理 ID
#             batch_list.extend([patient_idx] * num_features)
#
#         # 合并所有患者的节点特征和边
#         all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
#         all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)
#
#         all_edge_indices = torch.cat(edge_index_list, dim=1)
#         batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息
#
#         # 使用 Batch 来将图数据合并，保持不同患者的独立性
#         graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)
#
#         return graph_data
#
#     # 创建图数据
#     graph_data = create_graph_from_features(combined_features)
#
#     # 检查 x 的维度
#     print(f"x shape: {graph_data.x.shape}")
#     print(f"edge_index shape: {graph_data.edge_index.shape}")
#
#     # 定义图神经网络模型
#     class GNN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(GNN, self).__init__()
#             self.conv1 = GCNConv(in_channels, 128)
#             self.conv2 = GCNConv(128, 64)
#             self.conv3 = GCNConv(64, out_channels)
#
#         def forward(self, x, edge_index,batch):
#             x = self.conv1(x, edge_index)
#             x = F.relu(x)
#             x = self.conv2(x, edge_index)
#             x = F.relu(x)
#             x = self.conv3(x, edge_index)
#             return x
#
#
#     # 初始化模型
#     model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
#     data = graph_data.to(device)
#
#     # 使用图神经网络进行节点特征聚合
#     model.eval()
#     with torch.no_grad():
#         aggregated_features = model(data.x, data.edge_index, data.batch).cpu().numpy()
#
#     return index, aggregated_features, labels

#
# def main():
#     index, gnn_excel_feature, labels = gnn_extract_excel_features(
#         '/tmp/pycharm_project_357/HER2_excel_data/HER2-data.xlsx')
#     print(f"Patient IDs: {index[:10]}")
#     print(f"Aggregated Features: {gnn_excel_feature[:10]}")
#     print(f"Labels: {labels[:10]}")
#
#
# if __name__ == "__main__":
#     main()

# ##########K近邻###############
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.neighbors import NearestNeighbors
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# from sklearn.preprocessing import StandardScaler
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def add_self_loops(edge_index, num_nodes):
#     """
#     添加自环边（每个节点都和自己有一条边连接）
#     """
#     loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
#     edge_index = torch.cat([edge_index, loop_index], dim=1)
#     return edge_index
#
#
# def gnn_extract_excel_features(filename, k=5):
#     # 读取Excel数据
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#
#     # 患者 ID 和标签
#     index = readbook.iloc[:, 0].to_numpy()  # 患者ID
#     labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）
#
#     # 临床病理特征（去掉患者ID和标签）
#     features_df = readbook.iloc[:, 1:-1]
#
#     # 打印列名和特征数量
#     print(f"Feature columns: {features_df.columns}")
#     print(f"Number of features: {features_df.shape[1]}")
#
#     # 分析数值特征和类别特征
#     numeric_features = features_df.select_dtypes(include=[np.number])
#     categorical_features = features_df.select_dtypes(exclude=[np.number])
#
#     # 输出数值特征和类别特征的数量
#     print(f"Number of numeric features: {numeric_features.shape[1]}")
#     print(f"Number of categorical features: {categorical_features.shape[1]}")
#
#     # 对类别特征进行独热编码
#     if not categorical_features.empty:
#         categorical_features = pd.get_dummies(categorical_features)
#         print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")
#
#     # 合并数值特征和类别特征
#     combined_features = pd.concat([numeric_features, categorical_features], axis=1)
#     combined_features = combined_features.to_numpy(dtype=np.float32)
#     # 打印最终合并特征的数量
#     print(f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")
#
#     # 标准化特征数据
#     scaler = StandardScaler()
#     combined_features = scaler.fit_transform(combined_features)
#
#     # 使用K近邻计算邻居并构建图
#     def create_graph_from_features(features, k=5):
#         num_patients = features.shape[0]  # 患者数量
#         num_features = features.shape[1]  # 每个患者的特征数量（每列一个特征）
#
#         # 使用sklearn的NearestNeighbors计算每个点的k个最近邻
#         knn = NearestNeighbors(n_neighbors=k+1)  # +1 是为了排除节点与自己之间的连接
#         knn.fit(features)
#         distances, indices = knn.kneighbors(features)
#
#         edge_index_list = []  # 存储所有患者的边信息
#         node_features_list = []  # 存储所有患者的节点特征
#         batch_list = []  # 存储患者的批处理信息
#
#         for patient_idx in range(num_patients):
#             patient_features = features[patient_idx]  # 当前患者的特征
#             node_features_list.append(patient_features)
#
#             # 为每个患者设置一个独立的批处理 ID
#             batch_list.extend([patient_idx] * num_features)
#
#             # 对每个患者的邻居建立边
#             for neighbor_idx in indices[patient_idx][1:]:  # 排除自己
#                 edge_index_list.append([patient_idx, neighbor_idx])
#
#         # 合并所有患者的节点特征和边
#         all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
#         all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)
#
#         all_edge_indices = torch.tensor(np.array(edge_index_list).T, dtype=torch.long)  # 转换为PyTorch张量
#         batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息
#
#         # 使用 Batch 来将图数据合并，保持不同患者的独立性
#         graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)
#
#         return graph_data
#
#     # 创建图数据
#     graph_data = create_graph_from_features(combined_features, k=k)
#
#     # 检查 x 的维度
#     print(f"x shape: {graph_data.x.shape}")
#     print(f"edge_index shape: {graph_data.edge_index.shape}")
#
#     # 定义图神经网络模型
#     class GNN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(GNN, self).__init__()
#             self.conv1 = GCNConv(in_channels, 128)
#             self.conv2 = GCNConv(128, 64)
#             self.conv3 = GCNConv(64, out_channels)
#
#         def forward(self, x, edge_index, batch):
#             x = self.conv1(x, edge_index)
#             x = F.relu(x)
#             x = self.conv2(x, edge_index)
#             x = F.relu(x)
#             x = self.conv3(x, edge_index)
#             return x
#
#     # 初始化模型
#     model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
#     data = graph_data.to(device)
#
#     # 使用图神经网络进行节点特征聚合
#     model.eval()
#     with torch.no_grad():
#         aggregated_features = model(data.x, data.edge_index, data.batch).cpu().numpy()
#
#     return index, aggregated_features, labels

#
# # 余弦相似性
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# from sklearn.preprocessing import StandardScaler
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def add_self_loops(edge_index, num_nodes):
#     """
#     添加自环边（每个节点都和自己有一条边连接）
#     """
#     loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
#     edge_index = torch.cat([edge_index, loop_index], dim=1)
#     return edge_index
#
#
# def gnn_extract_excel_features(filename):
#     # 读取Excel数据
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#
#     # 患者 ID 和标签
#     index = readbook.iloc[:, 0].to_numpy()  # 患者ID
#     labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）
#
#     # 临床病理特征（去掉患者ID和标签）
#     features_df = readbook.iloc[:, 1:-1]
#
#     # 打印列名和特征数量
#     print(f"Feature columns: {features_df.columns}")
#     print(f"Number of features: {features_df.shape[1]}")
#
#     # 分析数值特征和类别特征
#     numeric_features = features_df.select_dtypes(include=[np.number])
#     categorical_features = features_df.select_dtypes(exclude=[np.number])
#
#     # 输出数值特征和类别特征的数量
#     print(f"Number of numeric features: {numeric_features.shape[1]}")
#     print(f"Number of categorical features: {categorical_features.shape[1]}")
#
#     # 对类别特征进行独热编码
#     if not categorical_features.empty:
#         categorical_features = pd.get_dummies(categorical_features)
#         print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")
#
#     # 合并数值特征和类别特征
#     combined_features = pd.concat([numeric_features, categorical_features], axis=1)
#     combined_features = combined_features.to_numpy(dtype=np.float32)
#     # 打印最终合并特征的数量
#     print(f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")
#
#     # 标准化特征数据
#     scaler = StandardScaler()
#     combined_features = scaler.fit_transform(combined_features)
#
#     # 使用余弦相似度计算邻居并构建图
#     def create_graph_from_features(features, similarity_threshold=0.8):
#         num_patients = features.shape[0]  # 患者数量
#         num_features = features.shape[1]  # 每个患者的特征数量（每列一个特征）
#
#         # 计算余弦相似度矩阵
#         similarity_matrix = cosine_similarity(features)
#
#         edge_index_list = []  # 存储所有患者的边信息
#         node_features_list = []  # 存储所有患者的节点特征
#         batch_list = []  # 存储患者的批处理信息
#
#         for patient_idx in range(num_patients):
#             patient_features = features[patient_idx]  # 当前患者的特征
#             node_features_list.append(patient_features)
#
#             # 为每个患者设置一个独立的批处理 ID
#             batch_list.extend([patient_idx] * num_features)
#
#             # 对每个患者的邻居建立边
#             for neighbor_idx in range(num_patients):
#                 if patient_idx != neighbor_idx and similarity_matrix[patient_idx, neighbor_idx] > similarity_threshold:
#                     edge_index_list.append([patient_idx, neighbor_idx])
#
#         # 合并所有患者的节点特征和边
#         all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
#         all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)
#
#         all_edge_indices = torch.tensor(np.array(edge_index_list).T, dtype=torch.long)  # 转换为PyTorch张量
#         batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息
#
#         # 使用 Batch 来将图数据合并，保持不同患者的独立性
#         graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)
#
#         return graph_data
#
#     # 创建图数据
#     graph_data = create_graph_from_features(combined_features, similarity_threshold=0.8)
#
#     # 检查 x 的维度
#     print(f"x shape: {graph_data.x.shape}")
#     print(f"edge_index shape: {graph_data.edge_index.shape}")
#
#     # 定义图神经网络模型
#     class GNN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(GNN, self).__init__()
#             self.conv1 = GCNConv(in_channels, 128)
#             self.conv2 = GCNConv(128, 64)
#             self.conv3 = GCNConv(64, out_channels)
#
#         def forward(self, x, edge_index, batch):
#             x = self.conv1(x, edge_index)
#             x = F.relu(x)
#             x = self.conv2(x, edge_index)
#             x = F.relu(x)
#             x = self.conv3(x, edge_index)
#             return x
#
#     # 初始化模型
#     model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
#     data = graph_data.to(device)
#
#     # 使用图神经网络进行节点特征聚合
#     model.eval()
#     with torch.no_grad():
#         aggregated_features = model(data.x, data.edge_index, data.batch).cpu().numpy()
#
#     return index, aggregated_features, labels
#

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_self_loops(edge_index, num_nodes):
    """
    添加自环边（每个节点都和自己有一条边连接）
    """
    loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


def gnn_extract_excel_features(filename):
    # 读取Excel数据
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')

    # 患者 ID 和标签
    index = readbook.iloc[:, 0].to_numpy()  # 患者ID
    labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）

    # 临床病理特征（去掉患者ID和标签）
    features_df = readbook.iloc[:, 1:-1]

    # 打印列名和特征数量
    print(f"Feature columns: {features_df.columns}")
    print(f"Number of features: {features_df.shape[1]}")

    # 分析数值特征和类别特征
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])

    # 输出数值特征和类别特征的数量
    print(f"Number of numeric features: {numeric_features.shape[1]}")
    print(f"Number of categorical features: {categorical_features.shape[1]}")

    # 对类别特征进行独热编码
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
        print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")

    # 合并数值特征和类别特征
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)
    # 打印最终合并特征的数量
    print(
        f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")

    # 构建图数据：每个患者为一个图，每个临床特征为一个节点
    def create_graph_from_features(features):
        num_patients = features.shape[0]  # 患者数量
        num_features = features.shape[1]  # 每个患者的临床特征数量（每列一个特征）

        edge_index_list = []  # 存储所有患者的边信息
        node_features_list = []  # 存储所有患者的节点特征
        batch_list = []  # 存储患者的批处理信息

        total_nodes = 0  # 记录所有患者的节点总数，用于批处理索引的映射

        for patient_idx in range(num_patients):
            patient_features = features[patient_idx]  # 当前患者的特征
            node_features_list.append(patient_features)

            # 构建边：基于特征间的关系
            edge_index = []
            for i in range(num_features):
                for j in range(num_features):
                    if i != j:  # 不创建自环，稍后会统一添加
                        edge_index.append([i, j])

            # 转换为 PyTorch 张量并添加自环
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_index = add_self_loops(edge_index, num_features)
            edge_index_list.append(edge_index)

            # 为每个患者设置一个独立的批处理 ID
            batch_list.extend([patient_idx] * num_features)
            total_nodes += num_features

        # 合并所有患者的节点特征和边
        all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
        all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)

        all_edge_indices = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息

        print(f"batch_tensor {batch_tensor}")
        print(f"num_patients {num_patients}")
        # 确保 batch_tensor 的索引没有超出范围
        max_batch_index = batch_tensor.max().item()
        if max_batch_index >= num_patients:
            raise ValueError(f"Batch index {max_batch_index} exceeds the number of patients ({num_patients})")

        # 使用 Batch 来将图数据合并，保持不同患者的独立性
        graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)

        return graph_data

    # 创建图数据
    graph_data = create_graph_from_features(combined_features)

    # 检查 x 的维度
    print(f"x shape: {graph_data.x.shape}")
    print(f"edge_index shape: {graph_data.edge_index.shape}")

    # 定义图神经网络模型
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
            # 使用图级池化（例如，全局平均池化）
            # pooled_features = global_mean_pool(x, batch)  # 对每个图进行池化
            return x  # 返回图级特征


    # 初始化模型
    model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
    data = graph_data.to(device)

    # 使用图神经网络进行节点特征聚合
    model.eval()
    with torch.no_grad():
        aggregated_features = model(data.x, data.edge_index, data.batch).cpu().numpy()

    return index, aggregated_features, labels
