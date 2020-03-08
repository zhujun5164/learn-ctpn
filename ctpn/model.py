import torch
import torch.nn as nn
import torchvision


class BasicConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channel, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class fc(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(fc, self).__init__()
        self.linear = nn.Linear(input_channel, output_channel)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return x


class predict_1_conv(nn.Module):
    def __init__(self, input_channel, anchor_size):
        super(predict_1_conv, self).__init__()
        self.predict_bbox = nn.Conv2d(input_channel, anchor_size * 4, kernel_size=1, stride=1)
        self.predict_cls = nn.Conv2d(input_channel, anchor_size * 2, kernel_size=1, stride=1)

    def forward(self, x):
        pre_bbox = self.predict_bbox(x)
        pre_cls = self.predict_cls(x)
        return pre_bbox, pre_cls


class predict_1_fc(nn.Module):
    def __init__(self, input_channel, anchor_size):
        super(predict_1_fc, self).__init__()
        self.predict_bbox = nn.Linear(input_channel, anchor_size * 4)
        self.predict_cls = nn.Linear(input_channel, anchor_size * 2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(-1, C)
        pre_bbox = self.predict_bbox(x)
        pre_cls = self.predict_cls(x)

        pre_bbox = pre_bbox.view(B, -1, H, W)
        pre_cls = pre_cls.view(B, -1, H, W)

        return pre_bbox, pre_cls
        

class ctpn(nn.Module):
    def __init__(self, feature_size, hidden_size, anchor_size):
        super(ctpn, self).__init__()
        self.anchor_size = anchor_size

        backbone = torchvision.models.vgg16(pretrained=True)
        backbone_layer = list(backbone.feature)[:-1]
        # [batch_size, 512, h_out, w_out]
        self.backbone = nn.Sequential(backbone_layer)
        # [batch_size, 512, h_out, w_out] -> [batch_size * h_out, 512, w_out]
        self.conv_1 = BasicConv(feature_size, feature_size, 3)
        self.Bilstm = nn.LSTM(feature_size, hidden_size, bias=True, batch_first=True, bidirectional=True)
        # [batch_size * h_out, 512, w_out]  batch_size, hidden_size, seq_len
        self.fc_lstm = fc(hidden_size * 2, feature_size)
        self.predict_1 = predict_1_fc(feature_size, anchor_size)
        self.softmax_2d = nn.Softmax2d()

    def forward(self, img):
        # img: [batch_size, 3, h, w]
        feature_map = self.backbone(img)
        # 经过一次卷积 + bn + relu，对feature_map进行处理
        feature_map = self.conv_1(feature_map)
        # [batch_size, 512, h_out, w_out]
        
        B, C, H, W = feature_map.shape
        feature_map = feature_map.view(-1, C, W)
        # [batch_size * h_out, 512, w_out]

        output, _ = self.Bilstm(feature_map)
        # output: batch_size * h_out, w_out, 2 * 128  [batch_size, seq_len, hidden_size * num_direction] 
        # hidden, cell: batch_size * h_out, 1 * 2, 128 [batch_size, num_layer * num_direction, hidden_size * num_direction] 

        # 好久写lstm模型了，总结了下，
        # 输入的feature_map尺寸为[batch_size * h_out, 512, w_out] batch_size, hidden_size, seq_len
        # 在这里，其实是认为文本是以行的形式存在的，然后每一个高度的w_out认为是一个长度，经过LSTM后能得到128 * 2维的特征向量
        # 而output代表的就是这lstm输出的双向特征

        # 对output进行维度转化，应该是变回[batch_size, 512, h_out, w_out]
        output_lstm = self.fc_lstm(output)
        output_lstm = output_lstm.view(B, -1, H, W)

        bbox_pre, cls_pre = self.predict_1(output_lstm)
        # pre_bbox: [batch_size, anchor_size * 4, H_out, W_out]
        # pre_cls: [batch_size, anchor_size * 2, H_out, W_out]

        prob_pre = self.softmax_2d(cls_pre.view(B, 2, H, W * self.anchor_size))
        # pre_prob: [batch_size, 2, H_out, W_out * anchor_size]
        prob_pre = prob_pre.view(B, 2 * self.anchor_size, H, W)
        # pre_prob: [batch_size, anchor_size * 2, H_out, W_out]

        return bbox_pre, cls_pre, prob_pre, feature_map


def anchor_generator(img, feature_map):
    # img: B, 3, H, W
    # feature_map: B, C, H_out, W_out

    # 原先ctpn代码中由于使用的是vgg16代码，因此widths固定在16，
    # 但实际上widths的数值是根据feature_map与img在宽度上的缩放比例而决定的
    # heights /0.7，这样得到
    img_h, img_w = img.shape[-2:]
    feature_h, feature_w = feature_map[-2:]

    # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    # widths = [16]
    stride_w = img_w // feature_w
    stride_h = img_h // feature_h

    hs = [round(stride_h * 0.7), stride_h]
    for i in range(8):
        hs.append(round(hs[-1] / 0.7))

    hs = torch.tensor(hs)
    ws = torch.tensor([stride_w]).repeat(hs.shape)
    
    base_anchor = torch.stack([-hs, -ws, hs, ws], dim=-1) // 2

    # 生成feature_map每个格子的中心代表在img上的像素点
    center_h = torch.arange(0, feature_h) * stride_h
    center_w = torch.arange(0, feature_w) * stride_w
    center_h, center_w = torch.meshgrid(center_h, center_w)
    center = torch.stack([center_h, center_w, center_h, center_w], dim=-1)

    anchor = center.view(-1, 1, 4) + base_anchor.view(1, -1, 4)
    anchor = anchor.view(-1, 4)
    return torch.round(anchor)
