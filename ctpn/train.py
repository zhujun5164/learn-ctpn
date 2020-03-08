from model import ctpn, anchor_generator
import cv2
import torch
import numpy as np

feature_size = 512
hidden_size = 128
anchor_size = 10
RPN_POSITIVE_OVERLAP = 0.7
RPN_NEGATIVE_OVERLAP = 0.3
RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256

imgs = []
imgs.append(cv2.imread('1.jpg'))
imgs = torch.LongTensor(imgs).permute(0, 3, 1, 2).float()
# 正常来说，imgs是要经过一定的处理的

model = ctpn(feature_size, hidden_size, anchor_size)
bbox_pre, cls_pre, prob_pre, feature_map = model(imgs)
# bbox_pre: [batch_size, anchor_size * 4, H_out, W_out]
# cls_pre: [batch_size, anchor_size * 2, H_out, W_out]
# prob_pre: [batch_size, anchor_size * 2, H_out, W_out]


# 以下是trian的内容
anchor = anchor_generator(imgs, feature_map)
# anchor: [batch_size * H_out * W_out * anchor_size, 4]
# 4: [h_1, w_1, h_2, w_2]

# 对生成的anchor进行裁剪，减去宽范围不在图像中的
# 取height, weight
dim = anchor.dim()
anchor_h = anchor[..., 0::2]
anchor_w = anchor[..., 1::2]
height, width = imgs[-2:]
anchor_h = anchor_h.clamp(min=0, max=height)
anchor_w = anchor_w.clamp(min=0, max=width)
clipped_anchor = torch.stack((anchor_h, anchor_w), dim=dim)

# 初始化label
labels = torch.empty(clipped_anchor.shape[0])
labels.fill_(-1)

# 在faster_rcnn上是需要对小尺寸网格进行删除的，即面积为0的部分
hs, ws = clipped_anchor[:, 2] - clipped_anchor[:, 0], clipped_anchor[:, 3] - clipped_anchor[:, 1]
keep = (ws > 0) & (hs > 0)
keep = keep.nonzero().squeeze(1)

keep_anchors = clipped_anchor[keep, :]
# 计算keep_anchor与gt_box的重合度, IoU
# 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组


def get_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_iou(box_1, box_2):
    area_1 = get_area(box_1)
    area_2 = get_area(box_2)

    lt = torch.max(box_1[:, None, :2], box_2[:, :2])  # [N,M,2]
    rb = torch.min(box_1[:, None, 2:], box_2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area_1[:, None] + area_2 - inter)
    return iou

iou_keepanchor_with_gtboxes = get_iou(keep_anchors, gt_boxes)
# [num_keep_anchors, num_gt_boxes]

# 获得每个anchor中，与gt_box的iou最大的值以及对应下标
anchor_argmax_iou, anchor_argmax_iou_label = iou_keepanchor_with_gtboxes.max(dim=1)

# 找到每个gt_box中，最大的iou以及label
gt_argmax_iou, _ = iou_keepanchor_with_gtboxes.max(dim=0)
gt_argmax_iou_label = torch.where(iou_keepanchor_with_gtboxes == gt_argmax_iou)[0]


# 对keep_anchor进行标签
labels[gt_argmax_iou_label] = 1
labels[anchor_argmax_iou >= RPN_POSITIVE_OVERLAP] = 1
labels[anchor_argmax_iou < RPN_NEGATIVE_OVERLAP] = 0

# 进行正负样本采样

# 获取正样本数量 -> 128
num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
fg_idx = torch.where(labels == 1)[0]

if len(fg_idx) > num_fg:
    # 生成fg_idx的下标， 对fg_idx下标进行随机排序
    num_id = np.arange(fg_idx.shape[0])
    np.random.shuffle(num_id)
    # 将随机排序后的fg_idx的下标 num_fg之后的赋值为-1
    labels[fg_idx[num_fg:]] = -1

# 对负样本进行采样，如果负样本的数量太多的话
# 正负样本总数是256，限制正样本数目最多128，
# 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
bg_idx = np.where(labels == 0)[0]
if len(bg_idx) > num_bg:
    num_id = np.arange(bg_idx.shape[0])
    np.random.shuffle(num_id)
    # 将随机排序后的bg_idx的下标 num_fg之后的赋值为-1
    labels[bg_idx[num_fg:]] = -1

# 对label上好标签后，计算rpn-box值
