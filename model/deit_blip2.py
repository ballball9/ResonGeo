import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from torch.utils.data import DataLoader

from dataset.dataset import CustomDataset


class CrossAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, head_dim, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N1, C = x1.shape
        B2, N2, C2 = x2.shape
        qkv1 = self.qkv(x1).reshape(B, N1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.qkv(x2).reshape(B, N2, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        # Compute cross attention scores
        q1 = q1 * self.scale
        attn1 = (q1 @ k2.transpose(-2, -1))  # x1 queries with x2 keys
        attn2 = (q2 @ k1.transpose(-2, -1))  # x2 queries with x1 keys

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)

        # Compute cross attention outputs
        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N1, C)
        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N2, C)

        x1 = self.proj(x1)
        x2 = self.proj(x2)
        x1 = self.proj_drop(x1)
        x2 = self.proj_drop(x2)

        return x1, x2

class DeiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.drop_path = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DeiTModel(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads, num_blocks, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.num_classes = 2048
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1  # For class token
        self.patch_embed = nn.Linear(num_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)
        self.blocks = nn.Sequential(*[DeiTBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.cross_attention = CrossAttentionModule(
            dim=self.embed_dim,  # Assuming embed_dim is the dimension of the embeddings
            num_heads=4,  # Assuming num_heads is the number of attention heads in the model
            head_dim=self.embed_dim // 4,  # Calculate head_dim based on embed_dim and num_heads
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,  # Dropout rate for attention
            proj_drop=0.0  # Dropout rate for projection
        )

    def forward_features(self, x, x_):
        B = x.shape[0]
        # x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_ = torch.cat((cls_tokens, x_), dim=1)
        x = x + self.pos_embed
        x_ = x_ + self.pos_embed
        x = self.pos_drop(x)
        x_ = self.pos_drop(x_)
        # x = self.norm(x)
        #
        # x, x_ = self.cross_attention(x, x_)
        # x, x_ = self.cross_attention(x, x_)
        # x, x_ = self.cross_attention(x, x_)
        x, x_ = self.cross_attention(x, x_)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return (x[:, 0] + x[:, 1])/2

    def forward(self, x, x_):
        x = self.forward_features(x, x_)
        # x = self.norm(x)
        x = self.head(x)
        return x



@register_model
def deit_small_distilled_patch16_224v2(pretrained=True):
    model = DeiTModel(
        num_patches=258,
        embed_dim=1408,
        num_heads=4,
        num_blocks=4,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.1,
        attn_drop=0.1
    )
    model.default_cfg = _cfg()
    num_classes = 2048
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )

        checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5, 1)[:num_classes, :]
        checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
        checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5, 1)[:num_classes, :]
        checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
        # for name, param in checkpoint["model"].items():
        #     print(name)
        # for name_m, param_m in model.state_dict().items():
        #     # for name, param in checkpoint["model"].items():
        #     if name_m in checkpoint["model"]:
        #         # print(name_m)
        #         model.state_dict()[name_m].copy_(checkpoint["model"][name_m])
        for name_m, param_m in model.state_dict().items():
            name_m_temp = name_m
            if 'in_proj_weight' in name_m:
                name_m_temp = name_m.replace('in_proj_weight', 'qkv.weight')
            if 'in_proj_bias' in name_m:
                name_m_temp = name_m.replace('in_proj_bias', 'qkv.bias')
            if 'out_proj.weight' in name_m:
                name_m_temp = name_m.replace('out_proj.weight', 'proj.weight')
            if 'out_proj.bias' in name_m:
                name_m_temp = name_m.replace('out_proj.bias', 'proj.bias')
            if 'mlp.0' in name_m:
                name_m_temp = name_m.replace('mlp.0', 'mlp.fc1')
            if 'mlp.0' in name_m:
                name_m_temp = name_m.replace('mlp.2', 'mlp.fc2')
            # for name, param in checkpoint["model"].items():
            if name_m_temp in checkpoint["model"]:
                # print(name_m, name_m_temp)
                temp = model.state_dict()[name_m]
                temp2 = checkpoint["model"][name_m_temp]
                if len(temp.shape) == 1:
                    temp[:temp2.shape[0]] = temp2

                    # temp[temp2.shape[0]:temp.shape[0]] = temp2[:(temp.shape[0]- temp2.shape[0])]
                if len(temp.shape) == 2:
                    temp[:temp2.shape[0], :temp2.shape[1]] = temp2
                    # temp[temp2.shape[0]:temp.shape[0], temp2.shape[1]:temp.shape[1]] = temp2[:(temp.shape[0] - temp2.shape[0]), :(temp.shape[1] - temp2.shape[1])]
                if len(temp.shape) == 3:
                    temp[:temp2.shape[0], :temp2.shape[1], :temp2.shape[2]] = temp2
                    # temp[temp2.shape[0]:temp.shape[0], temp2.shape[1]:temp.shape[1], temp2.shape[2]:temp.shape[2]] = temp2[:(temp.shape[0] - temp2.shape[0]), :(temp.shape[1] - temp2.shape[1]), :(temp.shape[2] - temp2.shape[2])]
                if len(temp.shape) == 4:
                    temp[:temp2.shape[0], :temp2.shape[1], :temp2.shape[2], :temp2.shape[3]] = temp2
                    # temp[temp2.shape[0]:temp.shape[0], temp2.shape[1]:temp.shape[1], temp2.shape[2]:temp.shape[2], temp2.shape[3]:temp.shape[3]] = temp2[:(temp.shape[0] - temp2.shape[0]), :(temp.shape[1] - temp2.shape[1]), :(temp.shape[2] - temp2.shape[2]), :(temp.shape[3] - temp2.shape[3])]

                model.state_dict()[name_m].copy_(nn.Parameter(temp))
    return model

# if __name__ == "__main__":
#     base_model = deit_small_distilled_patch16_224v2
#     model = base_model(pretrained=True)
#
#     # Create a dummy input with batch size 4 and sequence length 257
#     dummy_input = torch.randn(4, 257, 1408)
#
#     # Forward pass through the model
#     output = model(dummy_input)
#     print(output.shape)  # Should print torch.Size([4, 1000])
#
#     # 使用示例
#     dict_file_path = '/data1/CZX/Vit_huge_mlp/satellite_embedding_qformer.txt'  # 替换为你的字典文件路径
#     batch_size = 4
#     shuffle = True
#
#     # 创建数据集对象
#     dataset = CustomDataset(dict_file_path)
#
#     # 创建DataLoader
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
#     # 在训练循环中使用DataLoader
#     for batch_idx, (features, idx) in enumerate(data_loader):
#         # Forward pass
#         features =torch.squeeze(features, dim=1)
#         output = model(features)
#         print(output.shape)
#         pass
