import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_image_features(image_filenames):
    # 使用预训练的ResNet提取局部特征
    resnet_model = models.resnet18(weights='DEFAULT')  # 可以换成其他版本的ResNet
    resnet_model.fc = torch.nn.Identity()  # 去掉ResNet的全连接层，只保留卷积部分

    # 使用预训练的ViT提取全局特征
    vit_model = models.vit_l_32(weights='DEFAULT')
    vit_model.fc = torch.nn.Identity()  # 去掉ViT的分类头

    # 定义图像预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 设置为训练模式并将模型移动到指定设备
    resnet_model.train()
    vit_model.train()
    resnet_model.to(device)
    vit_model.to(device)

    image_features = []
    with torch.set_grad_enabled(False):  # 禁用梯度计算，提高推理效率
        for filename in image_filenames:
            image = Image.open(filename).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            # 使用ResNet提取局部特征
            resnet_features = resnet_model(input_tensor)
            resnet_features = resnet_features.squeeze().cpu().detach().numpy()  # 移除batch维度

            # 使用ViT提取全局特征
            vit_features = vit_model(input_tensor)
            vit_features = vit_features.squeeze().cpu().detach().numpy()  # 移除batch维度

            # 将ResNet和ViT特征进行拼接
            combined_features = np.concatenate([resnet_features, vit_features])

            image_features.append(combined_features)

    return np.vstack(image_features)

