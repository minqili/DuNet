import os

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import cv2
from torchvision import transforms


class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample: int = 30, device='cuda'):

        """
        初始化 MemoryBank 类。
        Parameters:
            normal_dataset (torch.utils.data.Dataset): 正常数据集，用于构建记忆库。
            nb_memory_sample (int): 记忆库中的样本数量。
            device (str): 设备类型，'cpu' 或 'cuda'.
        """
        self.device = device  # 设备类型（CPU或GPU）
        # 内存库
        self.memory_information = {}
        # self.memory = []
        # 正常数据集
        self.normal_dataset = self._load_images(normal_dataset)
        # 内存库中保存的样本数量
        self.nb_memory_sample = nb_memory_sample

    def _load_images(self, dataset_path):
        # 获取目录中所有图像文件的路径
        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def update(self, feature_extractor):
        """
        通过抽取正常数据集中的特征，更新记忆库。
        Parameters:
            feature_extractor (torch.nn.Module): 用于提取特征的神经网络模型。
        """
        feature_extractor.eval()  # 设置特征提取器为评估模式
        # 定义样本索引
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)  # 随机打乱样本索引
        # 提取特征并将特征保存到内存库中
        with torch.no_grad():
            for i in range(min(self.nb_memory_sample, len(samples_idx))):
                  # 选择图像路径
                img_path = self.normal_dataset[samples_idx[i]]  # 这一块应该是提取的照片，不是文件夹
                # # 读取图像
                input_normal = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if input_normal is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                # # 转换为 RGB 格式
                input_normal = cv2.cvtColor(input_normal, cv2.COLOR_BGR2RGB)
                # 转换为张量并移动到设备
                input_normal = transforms.ToTensor()(input_normal).unsqueeze(0).to(self.device).float()
                # 提取特征
                features, *_ = feature_extractor(input_normal)  # modify

                # 将特征保存到内存库中
                for j, features_l in enumerate(features[1:-1]):  # 避免使用相同的变量名 `i`
                    if f'level{j}' not in self.memory_information.keys():
                        self.memory_information[f'level{j}'] = features_l
                    else:
                        self.memory_information[f'level{j}'] = torch.cat(
                            [self.memory_information[f'level{j}'], features_l], dim=0)

    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        计算输入特征与记忆库中正常特征之间的差异。
        Parameters:
            features (List[torch.Tensor]): 要计算差异的特征列表。
        Returns:
            torch.Tensor: 特征差异矩阵。
        """
        # 批次大小 X 内存中保存的样本数量 确保 features 列表中包含张量
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # 各级特征
        for l, level in enumerate(self.memory_information):
            # 批次
            for b_idx, features_b in enumerate(features[l]):
                # 计算 L2 损失  # features 是输入的特征张量
                # 确保 target 的尺寸与 input 的尺寸匹配
                diff = F.mse_loss(
                    input=torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target=self.memory_information[level],
                    reduction='none'
                ).mean(dim=[1, 2, 3])

                # 累加损失
                diff_bank[b_idx] += diff

        return diff_bank

    def select(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        选择特征，将其与记忆库中具有最小差异的特征连接起来。
        Parameters:
            features (List[torch.Tensor]): 要选择的特征列表。
        Returns:
            torch.Tensor: 连接后的特征。
        """
        # 计算输入特征与内存库中正常特征的差异
        diff_bank = self._calc_diff(features=features)

        # 将输入特征与内存库中最小差异的特征进行拼接
        for l, level in enumerate(self.memory_information):
            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))

            # 调整大小 modify
            # 调整大小和通道数
            selected_features = F.interpolate(selected_features, size=features[l].size()[2:],
                                              mode='bilinear', align_corners=False)
            selected_features = selected_features[:, :features[l].size(1)]
            # end modify
            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)

            # diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            # features[l] = torch.cat([features[l], diff_features], dim=1)

        return features


    # def select(self, features: List[torch.Tensor]) -> list[torch.Tensor]:
    #
    #     # 计算特征与记忆库中正常特征之间的差异
    #     diff_bank = self._calc_diff(features=features)
    #
    #     # 将特征与记忆库中最小差异的特征连接起来
    #     for l, level in enumerate(self.memory_information.keys()):
    #         selected_features = torch.index_select(self.memory_information[level], dim=0,
    #                                                index=diff_bank.argmin(dim=1))
    #         diff_features = F.mse_loss(selected_features, features[l], reduction='none')
    #         features[l] = torch.cat([features[l], diff_features], dim=1)
    #
    #     return features




