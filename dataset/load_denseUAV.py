import ast
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torchvision.transforms as transforms


def input_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size=size, interpolation=3),  # 调整图像大小到256x256
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])


class FeatureDataset(Dataset):
    def __init__(self, model, processor, root='/data1/CZX/TransGeo/data/DenseUAV/train/drone'):
        self.dict_path = {}
        self.root = root
        self.model = model
        self.processor = processor
        self.size = [256, 256]  # [512, 512]
        self.transform_input = input_transform(size=self.size)  # Drone

        for cls_name in os.listdir(os.path.join(root)):
            # img_list = os.listdir(os.path.join(root, cls_name))
            # img_path_list = [os.path.join(root, cls_name, img) for img in img_list]
            # img_list = os.listdir(os.path.join(root, cls_name))
            # img_path_list = os.path.join(root, cls_name, 'H80.JPG')
            img_list = os.listdir(os.path.join(root, cls_name))
            img_path_list = [os.path.join(root, cls_name, img) for img in img_list]
            self.dict_path[cls_name] = img_path_list
        self.cls_names = os.listdir(os.path.join(root))
        self.cls_names.sort()
        # self.map_dict = {i: self.cls_names[i] for i in range(len(self.cls_names))}

    def __len__(self):
        return len(self.dict_path)

    def __getitem__(self, idx):
        # feature_merge = []
        # feature_merge = np.zeros([1, 197, 768])
        # feature_merge2 = []
        key = self.cls_names[idx]
        path = self.dict_path[key]
        path = np.random.choice(path, 1)[0]
        image = Image.open(path)
        # image = self.transform_input(image)
        inputs = self.processor(images=image, text='', return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        feature_pooling = outputs.vision_outputs.pooler_output.detach().cpu().numpy().tolist()
        feature_vector = outputs.vision_outputs.last_hidden_state.detach().cpu().numpy().tolist()
        # feature_vector = outputs.qformer_outputs.last_hidden_state.detach().cpu().numpy().tolist()
        # feature_q_pooling = outputs.qformer_outputs.pooler_output.detach().cpu().numpy().tolist()

        feature_pooling = torch.tensor(feature_pooling, dtype=torch.float32).unsqueeze(1)
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        # feature_q_pooling = torch.tensor(feature_q_pooling, dtype=torch.float32).unsqueeze(1)

        feature_vector = torch.cat((feature_vector, feature_pooling), dim=1)
        # feature_merge = torch.cat((feature_vector[:, :133, :768], (feature_vector[:, 133:195, :768] + feature_vector[:, 195:257, :768])/2), dim=1)
        # feature_merge2 = torch.cat((feature_vector[:, :133, 640:1408], (feature_vector[:, 133:195, 640:1408] + feature_vector[:, 195:257, 640:1408])/2), dim=1)
        # feature_final = torch.cat((feature_merge+feature_merge2, feature_pooling[:, :, :768], feature_pooling[:, :, 640:1408]), dim=1)
        # feature_merge[:, :32, :] = feature_vector
        # feature_merge[:, 32:64, :] = feature_vector
        # feature_merge[:, 64:96, :] = feature_vector
        # feature_merge[:, 96:128, :] = feature_vector
        # feature_merge[:, 128:160, :] = feature_vector
        # feature_merge[:, 160:192, :] = feature_vector
        # feature_merge[:, 192:197, :] = feature_q_pooling.double()
        # feature_merge[:, 195, :] = feature_pooling[:, :, :768]
        # feature_merge[:, 196, :] = feature_pooling[:, :, 640:1408]

        if feature_vector is not None:
            # return torch.tensor(feature_merge), torch.tensor(idx)
            return torch.tensor(feature_vector), torch.tensor(idx)
        else:
            raise IndexError(f"No feature vector found for index {idx}")


class FeatureDatasetSatellite(Dataset):
    def __init__(self, model, processor, root='/data1/CZX/TransGeo/data/DenseUAV/train/satellite'):
        self.dict_path = {}
        self.root = root
        self.model = model
        self.processor = processor

        for cls_name in os.listdir(os.path.join(root)):
            # img_list = os.listdir(os.path.join(root, cls_name))
            # img_path_list = os.path.join(root, cls_name, 'H80.tif')
            # img_list = os.listdir(os.path.join(root, cls_name, '150'))
            img_list = os.listdir(os.path.join(root, cls_name))
            img_path_list = [os.path.join(root, cls_name, img) for img in img_list]
            # img_path_list = [os.path.join(root, cls_name, '150', img) for img in img_list]
            self.dict_path[cls_name] = img_path_list
        self.cls_names = os.listdir(os.path.join(root))
        # self.cls_names.sort()
        # self.map_dict = {i: self.cls_names[i] for i in range(len(self.cls_names))}

    def __len__(self):
        return len(self.dict_path)

    def __getitem__(self, idx):
        # feature_merge = []
        feature_merge = np.zeros([1, 197, 768])
        # feature_merge2 = []
        key = self.cls_names[idx]
        path = self.dict_path[key]
        path = np.random.choice(path, 1)[0]
        image = Image.open(path)
        # image = self.transform_input(image)
        inputs = self.processor(images=image, text='', return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        feature_pooling = outputs.vision_outputs.pooler_output.detach().cpu().numpy().tolist()
        feature_vector = outputs.vision_outputs.last_hidden_state.detach().cpu().numpy().tolist()
        # feature_vector = outputs.qformer_outputs.last_hidden_state.detach().cpu().numpy().tolist()
        # feature_q_pooling = outputs.qformer_outputs.pooler_output.detach().cpu().numpy().tolist()

        feature_pooling = torch.tensor(feature_pooling, dtype=torch.float32).unsqueeze(1)
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        # feature_q_pooling = torch.tensor(feature_q_pooling, dtype=torch.float32).unsqueeze(1)

        feature_vector = torch.cat((feature_vector, feature_pooling), dim=1)


        if feature_vector is not None:
            # return torch.tensor(feature_merge), torch.tensor(idx)
            return torch.tensor(feature_vector), torch.tensor(idx)
        else:
            raise IndexError(f"No feature vector found for index {idx}")

class FeatureDroneDataset(Dataset):
    def __init__(self, model, processor,index=54, root='/data1/CZX/TransGeo/data/train/drone'):
        dict_path = {}
        self.root = root
        self.model = model
        self.processor = processor
        self.index = index

        for cls_name in os.listdir(os.path.join(root)):
            img_list = os.listdir(os.path.join(root, cls_name))
            img_path_list = [os.path.join(root, cls_name, img) for img in img_list]
            dict_path[cls_name] = img_path_list
        cls_names = os.listdir(os.path.join(root))
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}
        self.cls_test_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path

    def __len__(self):
        return len(self.dict_path)

    def __getitem__(self, idx):
        key = self.cls_names[idx]
        path = self.dict_path[key]
        feature_v = []
        for i in range(self.index):
            path_i = path[i]
            # path = np.random.choice(path, 1)[0]
            image = Image.open(path_i)
            # image = self.transform_input(image)
            inputs = self.processor(images=image, text='', return_tensors="pt").to(self.model.device)
            outputs = self.model(**inputs)
            # img_drone = outputs.vision_outputs.pooler_output
            feature_vector = outputs.vision_outputs.pooler_output.detach().cpu().numpy().tolist()
            feature_v.append(feature_vector)
        # feature_vector = outputs.qformer_outputs.last_hidden_state.detach().cpu().numpy().tolist()
        if feature_vector is not None:
            return torch.tensor(feature_v), torch.tensor(idx)
        else:
            raise IndexError(f"No feature vector found for index {idx}")

# # # #
# # # 使用示例
# batch_size = 4
# shuffle = True
# model_dir = "/data1/CZX/blip2-opt-2.7b"
# # Load the pre-trained BLIP-2 model and processor
# # 设置设备，如果可用的话使用GPU
# device_blip = "cuda" if torch.cuda.is_available() else "cpu"
# device_blip = 'cpu'
# # 从Hugging Face Hub加载预训练的BLIP-2模型和处理器
# processor_blip = Blip2Processor.from_pretrained(model_dir)
# model_blip = Blip2ForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32)
# model_blip.to(device_blip)
# # 创建数据集对象
# dataset = FeatureDataset(model_blip, processor_blip, root='/data1/CZX/TransGeo/data/train/drone')
# # 创建DataLoader
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
# # 在训练循环中使用DataLoader
# for batch_idx, (features,idx) in enumerate(data_loader):
#     # 在这里使用features，它是一个批量的特征向量
#     print(batch_idx, idx, features.shape)
#     pass
