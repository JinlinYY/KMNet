import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # (B, H)
        scores = self.V(energy)         # (B, 1)
        weights = F.softmax(scores, dim=1)  # (B, 1)
        return weights * x              # 加权输出 (B, D)

class GatedFusion(nn.Module):
    def __init__(self, dim1, dim2, fused_dim):
        super(GatedFusion, self).__init__()
        self.proj1 = nn.Linear(dim1, fused_dim)
        self.proj2 = nn.Linear(dim2, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1_ = self.proj1(x1)
        x2_ = self.proj2(x2)

        print("x1_", x1_.shape)
        print("x2_", x2_.shape)

        gate_input = torch.cat([x1_, x2_], dim=1)
        gate_weight = self.gate(gate_input)
        return gate_weight * x1_ + (1 - gate_weight) * x2_

class AttentionGateFusion(nn.Module):
    def __init__(self, img_dim, gnn_dim, hidden_dim=128):
        super().__init__()
        self.att_img = SelfAttention(img_dim, hidden_dim)
        self.att_gnn = SelfAttention(gnn_dim, hidden_dim)
        self.gate_final = GatedFusion(img_dim, gnn_dim, hidden_dim)

    def forward(self, img, gnn):
        img_att = self.att_img(img)
        gnn_att = self.att_gnn(gnn)
        fused = self.gate_final(img_att, gnn_att)
        return fused

def combine_features(image_features, tabular_features, gnn_features, batch_size=128):
    image_features = image_features.clone().detach().to(device)
    tabular_features = tabular_features.clone().detach().to(device)
    gnn_features = gnn_features.clone().detach().to(device)

    model = AttentionGateFusion(
        img_dim=image_features.shape[1],
        gnn_dim=gnn_features.shape[1]
    ).to(device)

    model.eval()
    all_outputs = []

    with torch.no_grad():
        for i in range(0, image_features.shape[0], batch_size):
            img_batch = image_features[i:i+batch_size]
            tab_batch = tabular_features[i:i+batch_size]
            gnn_batch = gnn_features[i:i+batch_size]

            fused = model(img_batch, gnn_batch)  # 只融合图像和图
            out = torch.cat([fused, tab_batch], dim=1)  # 表格特征直接拼接
            all_outputs.append(out.cpu())

    return torch.cat(all_outputs, dim=0).numpy()
