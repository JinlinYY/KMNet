#################HER2的超声图像特征提取#####################
# import numpy as np
# import torch
# from torchvision import models, transforms
# from PIL import Image
# from component.Cli_Encoder import extract_excel_features
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def extract_image_features(image_filenames):
#     model = models.vit_l_32(weights='DEFAULT')
#     # model = models.densenet121(weights='DEFAULT')
#     model.fc = torch.nn.Identity()
#
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     model.train()
#     model.to(device)
#
#     image_features = []
#     with torch.set_grad_enabled(False):  # Disable gradients
#         for filename in image_filenames:
#             image = Image.open(filename).convert('RGB')
#             input_tensor = preprocess(image).unsqueeze(0).to(device)
#             features = model(input_tensor)
#             features = features.squeeze().cpu().detach().numpy()  # Squeeze to remove batch dimension
#             image_features.append(features)
#
#     return np.vstack(image_features)


# def extract_image_features(image_filenames):
#     # 使用预训练的ResNet提取局部特征
#     resnet_model = models.resnet18(weights='DEFAULT')  # 可以换成其他版本的ResNet
#     resnet_model.fc = torch.nn.Identity()  # 去掉ResNet的全连接层，只保留卷积部分
#
#     # 使用预训练的ViT提取全局特征
#     vit_model = models.vit_l_32(weights='DEFAULT')
#     vit_model.fc = torch.nn.Identity()  # 去掉ViT的分类头
#
#     # 定义图像预处理步骤
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     # 设置为训练模式并将模型移动到指定设备
#     resnet_model.train()
#     vit_model.train()
#     resnet_model.to(device)
#     vit_model.to(device)
#
#     image_features = []
#     with torch.set_grad_enabled(False):  # 禁用梯度计算，提高推理效率
#         for filename in image_filenames:
#             image = Image.open(filename).convert('RGB')
#             input_tensor = preprocess(image).unsqueeze(0).to(device)
#
#             # 使用ResNet提取局部特征
#             resnet_features = resnet_model(input_tensor)
#             resnet_features = resnet_features.squeeze().cpu().detach().numpy()  # 移除batch维度
#
#             # 使用ViT提取全局特征
#             vit_features = vit_model(input_tensor)
#             vit_features = vit_features.squeeze().cpu().detach().numpy()  # 移除batch维度
#
#             # 将ResNet和ViT特征进行拼接
#             combined_features = np.concatenate([resnet_features, vit_features])
#
#             image_features.append(combined_features)
#
#     return np.vstack(image_features)

#################ISIC的超声图像特征提取#####################
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
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

def extract_image_features(image_filenames, batch_size=16, num_workers=4):
    model = models.vit_l_32(weights='DEFAULT')
    model.fc = torch.nn.Identity()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.train()
    model.to(device)

    # Create a dataset and DataLoader for batching and multi-threading
    dataset = ImageDataset(image_filenames, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    image_features = []

    with torch.no_grad():  # Disable gradients for inference
        for inputs in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            features = features.cpu().detach().numpy()  # Move the features back to CPU and convert to numpy
            image_features.append(features)

    return np.vstack(image_features)