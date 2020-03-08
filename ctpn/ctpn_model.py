import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
import code

#假设有一个图像数据
imgs = []
imgs.append(cv2.imread('1.jpg'))
imgs = torch.LongTensor(imgs).permute(0, 3, 1, 2).float()

# 模型结构
# vgg16（pretrained)
# 获取得到vgg16的features层以上模型
backbone = torchvision.models.vgg16(pretrained=True).features
# 将img输入模型中，获取feature_map
feature_maps = backbone(imgs)
feature_maps = list(feature_maps.values())
# 输出的feature应该为 batch_size, out_channel, H_out, W_out

# 制作图片输入到模型的transform

# 制作rpn层
# 生成rpn_head
class RPNhead(nn.Module):

    def __init__(self, in_channels, num_anchors):
        super(RPNhead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors, kernel_size=1, stride=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        # init parameters
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # 输入的是feature_map, feature_map有可能是多层(resnet)
        # x: C * [batch_size, out_channel, H_out, W_out]
        logits = []
        bbox_reg = []
        for feature in x:
            t = nn.ReLU(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return bbox_reg, logits

# 获取input_channel(不同模型下，input_channel不一样)
input_channel = 512

# 因为模型需要输入每个feature_map的点所生成的Anchor数目，所以要先定义下Anchor_generator的参数
Anchor_sizes = (16, 64, 128)
aspect_ratios = (0.5, 1, 2)

rpn_head = RPNhead(input_channel, len(Anchor_sizes) * len(aspect_ratios))
bbox_reg, logits = rpn_head(feature_maps)
# bbox_reg：预测的框与Anchor的相对坐标  C * [batch_size, num_Anchor * 4, h_out, w_out]
# logits: 预测的这个框存在检测目标的概率 C * [batch_size, num_Anchor, h_out, w_out]

# 这是由于，若我们预测的是在图像中的绝对位置，会因为数值的尺度跨度过大，这样预测的话会很不好
# 所以我们需要返回到预测的目标在feature_map中的坐标，这样的话我们要把Anchor生成出来
class Anchor_generator(nn.Module):

    def __init__(self, Anchor_sizes = (64, 256, 512), aspect_ratios = (0.5, 1.0, 2.0)):
        super(Anchor_generator, self).__init__()

        self.Anchor_sizes = Anchor_sizes
        self.aspect_ratios = aspect_ratios

    # 生成Anchor的话，需要原输入图片的img跟feature_map
    def forward(self, imgs, feature_maps):
        # imgs [batch_size, 3, H, W]
        # feature_maps C * [batch_size, out_channel, h_out, w_out]

        # 获取图片的尺寸
        img_sizes = imgs.shape[-2:]
        # 获取每一层feature_map的尺寸
        features_size = tuple([feature_map.shape[-2:] for feature_map in feature_maps])
        # 计算一个feature_map的特征所代表图像的像素点个数
        stride = [[img_sizes[0] / f[0], img_sizes[1] / f[1]] for f in features_size]

        # 生成base_anchor
        Anchor_sizes = torch.as_tensor(self.Anchor_sizes, dtype=torch.float32)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=torch.float32)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        # 所有ratio对各个Anchor_size做计算
        hs = (h_ratios[:, None] * Anchor_sizes[None, :]).view(-1)
        ws = (w_ratios[:, None] * Anchor_sizes[None, :]).view(-1)

        # hs, ws做拼接 N * 4, round 取整
        base_anchor = (torch.stack([-hs, -ws, hs, ws], dim=1) / 2).round()

        # 生成每个feature_map层的中心点, 获取每层的尺寸和步长stride
        anchors = []
        for f, s in zip(features_size, stride):
            # 生成h_c, w_c点
            feature_h, feature_w = f
            stride_h, stride_w = s
            h_c = torch.arange(0, feature_h, dtype=torch.float32) * stride_h
            w_c = torch.arange(0, feature_w, dtype=torch.float32) * stride_w
            h_c, w_c = torch.meshgrid(h_c, w_c)
            h_c = h_c.view(-1)
            w_c = w_c.view(-1)
            anchor = torch.stack([h_c, w_c, h_c, w_c]).view(-1, 1, 4) + \
                     base_anchor.view(1, -1, 4)

            anchors.append(anchor.view(-1, 4))
        return anchors

# 获取anchor
anchor_generator = Anchor_generator()
anchors = anchor_generator(imgs, feature_maps)

# 通过预测的pre_box与anhcor的相对位置，计算得到pre_box在输入图像上的绝对坐标
