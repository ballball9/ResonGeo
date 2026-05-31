import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        # 计算每个元素的标签，每三个元素的标签相同
        # labels = torch.floor(torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device) / 3)
        labels = labels.long()
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss, loss, loss



