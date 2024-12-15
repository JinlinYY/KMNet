import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        energy = torch.tanh(self.W(features))
        attention = F.softmax(self.V(energy), dim=1)
        context = attention * features
        return context

def combine_features(image_features, tabular_features, gnn_features):
    # #############ISIC#############
    # image_features = torch.tensor(image_features, dtype=torch.float32) # 转换为 torch.Tensor 并移到正确的设备上
    # tabular_features = torch.tensor(tabular_features, dtype=torch.float32)
    # gnn_features = torch.tensor(gnn_features, dtype=torch.float32)
    # ##############################
    # 对图像特征应用注意力机制
    attention_image = Attention(input_dim=image_features.shape[1], output_dim=image_features.shape[1], hidden_dim=64)
    attended_image_features = attention_image(image_features)

    # 对GNN特征应用注意力机制
    attention_gnn = Attention(input_dim=gnn_features.shape[1], output_dim=gnn_features.shape[1], hidden_dim=64)
    attended_gnn_features = attention_gnn(gnn_features)

    # Detach tensors before converting to numpy arrays
    attended_image_features_np = attended_image_features.detach().cpu().numpy()
    # attended_image_features_np = image_features.detach().cpu().numpy()
    # attended_tabular_features_np = attended_tabular_features.detach().cpu().numpy()
    attended_tabular_features_np = tabular_features.detach().cpu().numpy()
    attended_gnn_features_np = attended_gnn_features.detach().cpu().numpy()
    # attended_gnn_features_np = gnn_features.detach().cpu().numpy()
    # 将三种特征拼接在一起
    combined_features = np.concatenate(
        (attended_image_features_np, attended_tabular_features_np, attended_gnn_features_np), axis=1)

    return combined_features

###########两模态 直接拼接###################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_features_concat_2(image_features, tabular_features):
    # 直接将三种特征拼接在一起
    image_features_np = image_features.detach().cpu().numpy()
    tabular_features_np = tabular_features.detach().cpu().numpy()

    # 将三种特征拼接在一起
    combined_features = np.concatenate(
        (image_features_np, tabular_features_np), axis=1)

    return combined_features


###########三模态 直接拼接###################
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def combine_features(image_features, tabular_features, gnn_features):
#     # 直接将三种特征拼接在一起
#     image_features_np = image_features.detach().cpu().numpy()
#     tabular_features_np = tabular_features.detach().cpu().numpy()
#     gnn_features_np = gnn_features.detach().cpu().numpy()
#
#     # 将三种特征拼接在一起
#     combined_features = np.concatenate(
#         (image_features_np, tabular_features_np, gnn_features_np), axis=1)
#
#     return combined_features

# #####################RDF###################################
# class Attention(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(Attention, self).__init__()
#         self.W = nn.Linear(input_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, features):
#         energy = torch.tanh(self.W(features))
#         attention = F.softmax(self.V(energy), dim=1)
#         context = attention * features
#         return context
#
#
# def combine_features(image_features, tabular_features, gnn_features):
#     # 对图像特征应用注意力机制
#     attention_image = Attention(input_dim=image_features.shape[1], output_dim=image_features.shape[1], hidden_dim=64)
#     attended_image_features = attention_image(image_features)
#
#     # # 对表格特征应用注意力机制
#     # attention_tabular = Attention(input_dim=tabular_features.shape[1], output_dim=tabular_features.shape[1], hidden_dim=64)
#     # attended_tabular_features = attention_tabular(tabular_features)
#
#     # 对GNN特征应用注意力机制
#     attention_gnn = Attention(input_dim=gnn_features.shape[1], output_dim=gnn_features.shape[1], hidden_dim=64)
#     attended_gnn_features = attention_gnn(gnn_features)
#
#     # Detach tensors before converting to numpy arrays
#     attended_image_features_np = attended_image_features.detach().cpu().numpy()
#     # attended_image_features_np = image_features.detach().cpu().numpy()
#     # attended_tabular_features_np = attended_tabular_features.detach().cpu().numpy()
#     attended_tabular_features_np = tabular_features.detach().cpu().numpy()
#     attended_gnn_features_np = attended_gnn_features.detach().cpu().numpy()
#     # attended_gnn_features_np = gnn_features.detach().cpu().numpy()
#     # 将三种特征拼接在一起
#     combined_features = np.concatenate(
#         (attended_image_features_np, attended_tabular_features_np, attended_gnn_features_np), axis=1)
#
#     return combined_features

# #图像和临床均加入注意力后拼接
# class Attention(nn.Module):
#     def __init__(self, image_input_dim, tabular_input_dim, output_dim, hidden_dim):
#         super(Attention, self).__init__()
#         self.image_W = nn.Linear(image_input_dim, hidden_dim)
#         self.tabular_W = nn.Linear(tabular_input_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, image_features, tabular_features):
#         image_energy = torch.tanh(self.image_W(image_features))
#         tabular_energy = torch.tanh(self.tabular_W(tabular_features))
#
#         image_attention = F.softmax(self.V(image_energy), dim=1)
#         tabular_attention = F.softmax(self.V(tabular_energy), dim=1)
#
#         attended_image_features = image_attention * image_features
#         attended_tabular_features = tabular_attention * tabular_features
#
#         return attended_image_features, attended_tabular_features
#
# def combine_features(image_features, tabular_features):
#     attention = Attention(image_input_dim=image_features.shape[1], tabular_input_dim=tabular_features.shape[1],
#                           output_dim=image_features.shape[1], hidden_dim=100)
#     attended_image_features, attended_tabular_features = attention(image_features, tabular_features)
#
#     attended_image_features_np = attended_image_features.detach().cpu().numpy()
#     attended_tabular_features_np = attended_tabular_features.detach().cpu().numpy()
#
#     combined_features = np.concatenate((attended_image_features_np, attended_tabular_features_np), axis=1)
#     return combined_features


# class Attention(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(Attention, self).__init__()
#         self.W = nn.Linear(input_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, features):
#         energy = torch.tanh(self.W(features))
#         attention = F.softmax(self.V(energy), dim=1)
#         attended_features = attention * features
#         return attended_features
#
# class ResidualConnection(nn.Module):
#     def __init__(self, input_dim):
#         super(ResidualConnection, self).__init__()
#         self.fc = nn.Linear(input_dim, input_dim)
#
#     def forward(self, original_features, attended_features):
#         combined_features = self.fc(original_features) + attended_features
#         return combined_features
#
# def combine_features(image_features, tabular_features, hidden_dim=100):
#     attention = Attention(input_dim=image_features.shape[1],
#                           output_dim=image_features.shape[1],
#                           hidden_dim=hidden_dim)
#
#     attended_image_features = attention(image_features)
#     attended_tabular_features = attention(tabular_features)
#
#     residual_connection = ResidualConnection(input_dim=image_features.shape[1])
#     combined_features = residual_connection(image_features, attended_image_features) + \
#                         residual_connection(tabular_features, attended_tabular_features)
#
#     combined_features_np = combined_features.detach().cpu().numpy()
#     return combined_features_np
#################################


# # StructuredAttention 类用于表格数据，应用了简单的结构化注意力机制。
# # MultiLevelAttention 类用于图像数据，应用了多层级注意力机制。
# # MultiModalAttention 类用于整合两种模态的特征，将注意力机制应用于结合后的特征。
#
# class StructuredAttention(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(StructuredAttention, self).__init__()
#         self.W = nn.Linear(input_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, features):
#         energy = torch.tanh(self.W(features))
#         attention = F.softmax(self.V(energy), dim=1)
#         attended_features = attention * features
#         return attended_features
#
# class MultiLevelAttention(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(MultiLevelAttention, self).__init__()
#         self.image_W1 = nn.Linear(input_dim, hidden_dim)
#         self.image_W2 = nn.Linear(hidden_dim, hidden_dim)
#         self.V = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, image_features):
#         energy = torch.tanh(self.image_W1(image_features))
#         energy = torch.tanh(self.image_W2(energy))
#         attention = F.softmax(self.V(energy), dim=1)
#         attended_features = attention * image_features
#         return attended_features
#
# class MultiModalAttention(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim):
#         super(MultiModalAttention, self).__init__()
#         self.W1 = nn.Linear(input_dim1, output_dim)
#         self.W2 = nn.Linear(input_dim2, output_dim)
#         self.V = nn.Linear(output_dim, output_dim)
#
#     def forward(self, features1, features2):
#         energy1 = self.W1(features1)
#         energy2 = self.W2(features2)
#         attention = F.softmax(self.V(energy1 + energy2), dim=1)
#         attended_features1 = attention * features1
#         attended_features2 = attention * features2
#         return attended_features1, attended_features2
#
# def combine_features(image_features, tabular_features):
#     structured_attention = StructuredAttention(input_dim=tabular_features.shape[1],
#                                                output_dim=tabular_features.shape[1],
#                                                hidden_dim=100)
#     multi_level_attention = MultiLevelAttention(input_dim=image_features.shape[1],
#                                                 output_dim=image_features.shape[1],
#                                                 hidden_dim=100)
#     multi_modal_attention = MultiModalAttention(input_dim1=image_features.shape[1],
#                                                 input_dim2=tabular_features.shape[1],
#                                                 output_dim=image_features.shape[1])
#
#     attended_tabular_features = structured_attention(tabular_features)
#     attended_image_features = multi_level_attention(image_features)
#
#     combined_image_features, combined_tabular_features = multi_modal_attention(attended_image_features, attended_tabular_features)
#
#     combined_features = torch.cat((combined_image_features, combined_tabular_features), dim=1)
#     return combined_features.detach().cpu().numpy()

# ##########################TFN####################################################################
# class TFN(nn.Module):
#     def __init__(self, input_dim_image, input_dim_tabular, output_dim):
#         super(TFN, self).__init__()
#         self.fc_image = nn.Linear(input_dim_image, output_dim)
#         self.fc_tabular = nn.Linear(input_dim_tabular, output_dim)
#
#     def forward(self, image_features, tabular_features):
#         # Adding bias terms
#         image_features_with_bias = torch.cat(
#             [image_features, torch.ones(image_features.size(0), 1).to(image_features.device)], dim=1)
#         tabular_features_with_bias = torch.cat(
#             [tabular_features, torch.ones(tabular_features.size(0), 1).to(tabular_features.device)], dim=1)
#
#         # Outer product
#         outer_product = torch.bmm(
#             image_features_with_bias.unsqueeze(2),  # Shape: (batch_size, image_dim+1, 1)
#             tabular_features_with_bias.unsqueeze(1)  # Shape: (batch_size, 1, tabular_dim+1)
#         )  # Resulting shape: (batch_size, image_dim+1, tabular_dim+1)
#
#         fused_features = outer_product.view(image_features.size(0),
#                                             -1)  # Flatten to (batch_size, (image_dim+1) * (tabular_dim+1))
#         return fused_features
#
# def combine_features(image_features, tabular_features):
#     tfn = TFN(input_dim_image=image_features.shape[1], input_dim_tabular=tabular_features.shape[1], output_dim=64)
#     combined_features = tfn(image_features, tabular_features)
#     combined_features_np = combined_features.detach().cpu().numpy()
#     return combined_features_np
# ##############################################################################################




###########################多头注意力机制####################################################################################
# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, query, key, value):
#         attn_output, _ = self.multihead_attn(query, key, value)
#         return self.output_proj(attn_output)
#
# class FeatureFusion(nn.Module):
#     def __init__(self, image_dim, tabular_dim, embed_dim, num_heads):
#         super(FeatureFusion, self).__init__()
#         self.image_proj = nn.Linear(image_dim, embed_dim)
#         self.tabular_proj = nn.Linear(tabular_dim, embed_dim)
#         self.multihead_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, image_features, tabular_features):
#         image_proj = self.image_proj(image_features)
#         tabular_proj = self.tabular_proj(tabular_features)
#         combined = torch.cat((image_proj.unsqueeze(1), tabular_proj.unsqueeze(1)), dim=1)
#         attended_features = self.multihead_attention(combined, combined, combined).mean(dim=1)
#         return self.output_proj(attended_features)
#
# def combine_features(image_features, tabular_features):
#     fusion_model = FeatureFusion(image_dim=image_features.shape[1], tabular_dim=tabular_features.shape[1], embed_dim=64, num_heads=8)
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()


# ############################外积矩阵-双模态####################################################################################
# import torch
# import torch.nn as nn
#
# class OuterProductFusion(nn.Module):
#     def __init__(self):
#         super(OuterProductFusion, self).__init__()
#
#     def forward(self, image_features, tabular_features):
#         batch_size = image_features.size(0)
#         outer_product_matrix = torch.bmm(image_features.unsqueeze(2), tabular_features.unsqueeze(1))
#         return outer_product_matrix.view(batch_size, -1)
#
# def combine_features(image_features, tabular_features):
#     fusion_model = OuterProductFusion()
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()
# # # ############################外积矩阵-三模态####################################################################################
# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class OuterProductFusion(nn.Module):
#     def __init__(self):
#         super(OuterProductFusion, self).__init__()
#
#     def forward(self, image_features, tabular_features):
#         batch_size = image_features.size(0)
#         outer_product_matrix = torch.bmm(image_features.unsqueeze(2), tabular_features.unsqueeze(1))
#         return outer_product_matrix.view(batch_size, -1)
#
#
# def combine_features(image_features, excel_features, tabular_features):
#     fusion_model = OuterProductFusion()
#     fus_features = fusion_model(image_features, tabular_features)
#
#     # Detach the tensor and move to CPU, then convert to NumPy
#     fus_features_np = fus_features.detach().cpu().numpy()
#
#     # Convert excel_features to NumPy if it's a tensor
#     if isinstance(excel_features, torch.Tensor):
#         excel_features = excel_features.detach().cpu().numpy()
#
#     # Concatenate along the correct axis
#     fused_features = np.concatenate((fus_features_np, excel_features), axis=1)
#
#     return fused_features

############################双线性池化-双模态####################################################################################
# import torch
# import torch.nn as nn
#
# class BilinearPooling(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim):
#         super(BilinearPooling, self).__init__()
#         self.fc = nn.Linear(input_dim1 * input_dim2, output_dim)
#
#     def forward(self, image_features, tabular_features):
#         batch_size = image_features.size(0)
#         outer_product = torch.bmm(image_features.unsqueeze(2), tabular_features.unsqueeze(1)).view(batch_size, -1)
#         return self.fc(outer_product)
#
# def combine_features(image_features, tabular_features):
#     fusion_model = BilinearPooling(input_dim1=image_features.shape[1], input_dim2=tabular_features.shape[1], output_dim=128)
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()
# ###########################双线性池化-三模态####################################################################################
# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class BilinearPooling(nn.Module):
#     def __init__(self, input_dim1, input_dim2, output_dim):
#         super(BilinearPooling, self).__init__()
#         self.fc = nn.Linear(input_dim1 * input_dim2, output_dim)
#
#     def forward(self, image_features, tabular_features):
#         batch_size = image_features.size(0)
#         outer_product = torch.bmm(image_features.unsqueeze(2), tabular_features.unsqueeze(1)).view(batch_size, -1)
#         return self.fc(outer_product)
#
#
# def combine_features(image_features, excel_features, tabular_features):
#     fusion_model = BilinearPooling(input_dim1=image_features.shape[1], input_dim2=tabular_features.shape[1],
#                                    output_dim=128)
#     combined_features = fusion_model(image_features, tabular_features)
#
#     # Detach the tensor and move it to the CPU if necessary
#     combined_features = combined_features.detach().cpu().numpy()
#
#     # Convert excel_features to numpy if it's a tensor
#     if isinstance(excel_features, torch.Tensor):
#         excel_features = excel_features.detach().cpu().numpy()
#
#     fused_features = np.concatenate((combined_features, excel_features), axis=1)
#     return fused_features

# ############################共注意力机制-双模态##################################################################################
# import torch
# import torch.nn as nn
#
# class CoAttention(nn.Module):
#     def __init__(self, image_dim, tabular_dim, output_dim):
#         super(CoAttention, self).__init__()
#         # Calculate projection dimensions
#         self.image_fc = nn.Linear(image_dim, output_dim)
#         self.tabular_fc = nn.Linear(tabular_dim, output_dim)
#         self.output_proj = nn.Linear(output_dim * output_dim, output_dim)  # Update this line
#
#     def forward(self, image_features, tabular_features):
#         # Project image and tabular features
#         image_proj = self.image_fc(image_features)
#         tabular_proj = self.tabular_fc(tabular_features)
#
#         # Compute attention matrix and flatten
#         attention_matrix = torch.bmm(image_proj.unsqueeze(2), tabular_proj.unsqueeze(1))
#         co_attention = attention_matrix.view(image_features.size(0), -1)
#
#         # Apply linear projection
#         return self.output_proj(co_attention)
#
# def combine_features(image_features, tabular_features):
#     # Ensure dimensions match with the CoAttention implementation
#     fusion_model = CoAttention(image_dim=image_features.shape[1], tabular_dim=tabular_features.shape[1], output_dim=128)
#     fused_features = fusion_model(image_features, tabular_features)
#     return fused_features.detach().cpu().numpy()
# ############################共注意力机制-三模态##################################################################################
# import torch
# import torch.nn as nn
# import numpy as np
#
#
# class CoAttention(nn.Module):
#     def __init__(self, image_dim, tabular_dim, output_dim):
#         super(CoAttention, self).__init__()
#         # Linear layers to project image and tabular features
#         self.image_fc = nn.Linear(image_dim, output_dim)
#         self.tabular_fc = nn.Linear(tabular_dim, output_dim)
#         # Linear layer to project the flattened attention matrix to the output dimension
#         self.output_proj = nn.Linear(output_dim * output_dim, output_dim)
#
#     def forward(self, image_features, tabular_features):
#         # Project image and tabular features
#         image_proj = self.image_fc(image_features)
#         tabular_proj = self.tabular_fc(tabular_features)
#
#         # Compute the outer product to get the attention matrix
#         attention_matrix = torch.bmm(image_proj.unsqueeze(2), tabular_proj.unsqueeze(1))
#
#         # Flatten the attention matrix to (batch_size, output_dim * output_dim)
#         co_attention = attention_matrix.view(image_features.size(0), -1)
#
#         # Apply a linear projection to the flattened attention matrix
#         return self.output_proj(co_attention)
#
#
# def combine_features(image_features, excel_features, tabular_features):
#     # Ensure dimensions match with the CoAttention implementation
#     fusion_model = CoAttention(image_dim=image_features.shape[1], tabular_dim=tabular_features.shape[1], output_dim=128)
#     fused_features = fusion_model(image_features, tabular_features)
#
#     # Detach and convert fused features to a NumPy array
#     fused_features = fused_features.detach().cpu().numpy()
#
#     # Detach and convert excel_features to NumPy if it's a tensor
#     if isinstance(excel_features, torch.Tensor):
#         excel_features = excel_features.detach().cpu().numpy()
#
#     # Concatenate along the correct axis
#     fused_features = np.concatenate((fused_features, excel_features), axis=1)
#
#     return fused_features


