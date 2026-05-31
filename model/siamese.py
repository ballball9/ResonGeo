from torch import nn
from model.deit_blip2 import deit_small_distilled_patch16_224v2
from model.clip_vitb_CAN import deit_base_clip, deit_large_clip, deit_large_vit, deit_huge_vit


class SiameseTransformer(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseTransformer, self).__init__()
        base_model_reference = deit_small_distilled_patch16_224v2
        base_model_query = deit_small_distilled_patch16_224v2

        self.query_net = base_model_query(pretrained=True)  # 无人机
        self.reference_net = base_model_reference(pretrained=True)  # 卫星
        self.polar = None

    def forward(self, im_q, im_k):
        return self.query_net(im_q, im_k), self.reference_net(im_k, im_q)


class SiameseClipVitb(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseClipVitb, self).__init__()
        base_model_reference = deit_base_clip
        base_model_query = deit_base_clip

        self.query_net = base_model_query(pretrained=True)  # 无人机
        self.reference_net = base_model_reference(pretrained=True)  # 卫星
        self.polar = None

    def forward(self, im_q, im_k):

        return self.query_net(im_q, im_k), self.reference_net(im_k, im_q)

class SiameseClipVitLarge(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseClipVitLarge, self).__init__()
        base_model_reference = deit_large_clip
        base_model_query = deit_large_clip

        self.query_net = base_model_query(pretrained=True)  # 无人机
        self.reference_net = base_model_reference(pretrained=True)  # 卫星
        self.polar = None

    def forward(self, im_q, im_k):

        return self.query_net(im_q, im_k), self.reference_net(im_k, im_q)


class SiameseVitLarge(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseVitLarge, self).__init__()
        base_model_reference = deit_large_vit
        base_model_query = deit_large_vit

        self.query_net = base_model_query(pretrained=True)  # 无人机
        self.reference_net = base_model_reference(pretrained=True)  # 卫星
        self.polar = None

    def forward(self, im_q, im_k):

        return self.query_net(im_q, im_k), self.reference_net(im_k, im_q)


class SiameseVitHuge(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseVitHuge, self).__init__()
        base_model_reference = deit_huge_vit
        base_model_query = deit_huge_vit

        self.query_net = base_model_query(pretrained=True)  # 无人机
        self.reference_net = base_model_reference(pretrained=True)  # 卫星
        self.polar = None

    def forward(self, im_q, im_k):

        return self.query_net(im_q, im_k), self.reference_net(im_k, im_q)
