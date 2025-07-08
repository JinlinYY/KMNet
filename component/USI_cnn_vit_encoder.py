import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, image_filenames, transform=None):
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class CNN_ViT(nn.Module):
    def __init__(self):
        super(CNN_ViT, self).__init__()
        # 使用 DenseNet121 的前几层作为浅层CNN
        densenet = models.densenet121(pretrained=True).features
        self.cnn = nn.Sequential(*list(densenet.children())[:6])  # 输出约为 (B, 128, 28, 28)

        # 卷积降维到ViT输入通道数
        self.proj = nn.Conv2d(128, 768, kernel_size=1)

        # 初始化ViT
        vit = models.vit_b_16(weights='DEFAULT')  # 预训练 ViT
        self.vit_encoder = vit.encoder

        # 修改 ViT 的位置编码长度（28x28 patch + 1 cls_token = 785）
        self.vit_encoder.pos_embedding = nn.Parameter(torch.zeros(1, 785, 768))
        self.vit_encoder.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        nn.init.trunc_normal_(self.vit_encoder.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.vit_encoder.cls_token, std=0.02)

    def forward(self, x):
        x = self.cnn(x)  # (B, 128, 28, 28)
        x = self.proj(x)  # (B, 768, 28, 28)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 784, 768)

        cls_token = self.vit_encoder.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat((cls_token, x), dim=1)  # (B, 785, 768)

        x = x + self.vit_encoder.pos_embedding  # 加入位置编码
        x = self.vit_encoder(x)  # ViT Encoder
        return x[:, 0]  # 取 cls_token 表示整图特征

def extract_image_features(image_filenames, batch_size=16, num_workers=4):
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据
    dataset = ImageDataset(image_filenames, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)

    # 初始化模型
    model = CNN_ViT().to(device)
    model.eval()

    image_features = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)  # (B, 768)
            image_features.append(features.cpu().numpy())

    return np.vstack(image_features)

