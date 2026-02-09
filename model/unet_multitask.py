"""
多任务 U-Net: 同时做分割 + 分类
- 分割: 二分类 (前景/背景)
- 分类: 3 类 (动物类/植物类/复合类)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet_backbone import resnet50


class unetUp(nn.Module):
    """U-Net 解码模块（上采样模块）"""
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class MultiTaskUNet(nn.Module):
    """
    多任务 U-Net
    
    输出:
        - seg_logits: [B, 1, H, W] 分割 logits (前景/背景)
        - cls_logits: [B, num_classes] 分类 logits (动物/植物/复合)
    """
    def __init__(self, num_seg_classes=1, num_cls_classes=3, backbone='resnet50'):
        super(MultiTaskUNet, self).__init__()
        
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        
        # Encoder (共享)
        if backbone == 'resnet50':
            self.encoder = resnet50()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Decoder (分割分支)
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]
        
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 分割头: 输出 1 通道 (前景/背景)
        self.seg_head = nn.Conv2d(out_filters[0], num_seg_classes, kernel_size=1)
        
        # 分类头: 从 encoder 最后一层特征做分类
        # ResNet50 最后一层特征通道数是 2048
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 [B, 2048, 1, 1]
            nn.Flatten(),             # [B, 2048]
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_cls_classes)  # [B, num_cls_classes]
        )
        
    def forward(self, inputs):
        """
        Args:
            inputs: [B, 3, H, W]
        
        Returns:
            seg_logits: [B, 1, H, W] 分割 logits
            cls_logits: [B, num_cls_classes] 分类 logits
        """
        # Encoder 提取多尺度特征
        [feat1, feat2, feat3, feat4, feat5] = self.encoder.forward(inputs)
        
        # 分类分支: 使用最深的特征 feat5
        cls_logits = self.cls_head(feat5)  # [B, num_cls_classes]
        
        # 分割分支: Decoder
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.up_conv(up1)
        
        seg_logits = self.seg_head(up1)  # [B, 1, H, W]
        
        return seg_logits, cls_logits


class MultiTaskLoss(nn.Module):
    """
    多任务损失: 分割损失 + 分类损失
    """
    def __init__(self, seg_loss_fn=None, cls_loss_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.seg_loss_fn = seg_loss_fn or nn.BCEWithLogitsLoss()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.cls_loss_weight = cls_loss_weight
        
    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets):
        """
        Args:
            seg_logits: [B, 1, H, W]
            cls_logits: [B, num_cls_classes]
            seg_targets: [B, H, W] 分割标签 (0: 背景, 1: 前景)
            cls_targets: [B] 分类标签 (0: 动物类, 1: 植物类, 2: 复合类)
        
        Returns:
            total_loss, seg_loss, cls_loss
        """
        # 分割损失
        seg_loss = self.seg_loss_fn(seg_logits.squeeze(1), seg_targets.float())
        
        # 分类损失
        cls_loss = self.cls_criterion(cls_logits, cls_targets)
        
        # 总损失
        total_loss = seg_loss + self.cls_loss_weight * cls_loss
        
        return total_loss, seg_loss, cls_loss
