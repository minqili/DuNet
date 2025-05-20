import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import cv2
from torchvision import transforms


class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample: int = 30, device='cuda'):

        self.device = device 
        self.memory_information = {}
        self.normal_dataset = self._load_images(normal_dataset)
        self.nb_memory_sample = nb_memory_sample

    def _load_images(self, dataset_path):

        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def update(self, feature_extractor):

        feature_extractor.eval()  
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)  

        with torch.no_grad():
            for i in range(min(self.nb_memory_sample, len(samples_idx))):

                img_path = self.normal_dataset[samples_idx[i]]  

                input_normal = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if input_normal is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                input_normal = cv2.cvtColor(input_normal, cv2.COLOR_BGR2RGB)

                input_normal = transforms.ToTensor()(input_normal).unsqueeze(0).to(self.device).float()

                features, *_ = feature_extractor(input_normal) 

                for j, features_l in enumerate(features[1:-1]): 
                    if f'level{j}' not in self.memory_information.keys():
                        self.memory_information[f'level{j}'] = features_l
                    else:
                        self.memory_information[f'level{j}'] = torch.cat(
                            [self.memory_information[f'level{j}'], features_l], dim=0)

    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:

        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        for l, level in enumerate(self.memory_information):
            for b_idx, features_b in enumerate(features[l]):

                diff = F.mse_loss(
                    input=torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target=self.memory_information[level],
                    reduction='none'
                ).mean(dim=[1, 2, 3])

                diff_bank[b_idx] += diff

        return diff_bank

    def select(self, features: List[torch.Tensor]) -> torch.Tensor:

        diff_bank = self._calc_diff(features=features)


        for l, level in enumerate(self.memory_information):
            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))

            selected_features = F.interpolate(selected_features, size=features[l].size()[2:],
                                              mode='bilinear', align_corners=False)
            selected_features = selected_features[:, :features[l].size(1)]

            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)

        return features



