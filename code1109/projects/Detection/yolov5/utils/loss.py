# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss 解决数据不均衡以及训练困难时候的一种损失函数
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        # 针对判断是否有物体的损失对应的权重信息
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions List[Tensor] Tensor的shape形状[bs,na,h,w,c+5], targets [m,6]
        # class loss 分类损失  正样本计算 --> 当前anchor box对应属于实际物体类别的损失
        lcls = torch.zeros(1, device=self.device)
        # box loss 回归损失  正样本计算 --> 当前anchor box的CIoU Loss
        lbox = torch.zeros(1, device=self.device)
        # object loss 是否有物体的分类损失  所有anchor box均计算，负样本对应是实际标签就是0，正样本对应的实际标签就是样本和GT Box之间的CIoU值
        lobj = torch.zeros(1, device=self.device)

        # 构建anchor box先验框和真实边框之间的匹配信息 也就是提取正样本等匹配信息;
        # 返回结构每个对象都一个list，list中存储的是每一个/每一层feature map对应的匹配信息
        # tcls: 对应的实际边框类别id
        # tbox: 边框回归相关信息 [offset_x, offset_y, w, h]  offset_x/offset_y是中心点在feature map上相比于单元格左上角的偏移值
        # indices: list(tuple4(tensor))，里面主要存储索引位置信息: img_idx、anchor_idx、grid_y、grid_x
        # anchors: 给定先验框大小(这些匹配的先验框)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image 对应的GT box对应的图像id, anchor idx, gridy, gridx
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets 正样本的数量(有多少个anchor box有对应的GT box)
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression 边框回归损失
                pxy = pxy.sigmoid() * 2 - 0.5  # 得到预测边框中心相比于单元格左上角的偏移值
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box 预测边框的"坐标信息"
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness 构建anchor box是否有物体的真实标签值(iou)
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio 将预测框和真实边框的IoU作为分类的标签值(替代标签1)

                # Classification 物体分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # 分类损失 -> 计算anchor box是否有物体的损失
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h) -> [nt,6]
        # 如果一个真实边框的中心点落在某个单元格内，那么由当前单元格的预测框/anchor box对该真实边框进行匹配预测
        # anchor box --对应的实际标签信息--> GT Box
        # 选择高度和宽度比例在1/4~4之间的作为匹配anchor box
        # grid cell负责预测GT box中心点落在grid cell附近的box的匹配预测

        na, nt = self.na, targets.shape[0]  # number of anchors 每个grid cell对应box的数量, targets 真实边框的数目
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # [na,nt] na个box，nt是总的真实边框数量
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # [na,nt,7] 在targets的基础上，针对每个gt box增加一个anchor box index信息 image,class,x,y,w,h,anchor_box_idx
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],  # 真实边框中心点所落的当前单元格
                [1, 0],  # j 左侧单元格
                [0, 1],  # k 上方单元格
                [-1, 0],  # l 右侧单元格
                [0, -1],  # m 下方单元格
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):  # eg:3 针对每一层的anchor box进行匹配
            anchors, shape = self.anchors[i], p[i].shape  # 获取当前层的先验框尺度和Feature Map大小
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7) 获得真实边框在当前层的对应坐标以及尺度大小(Feautre Map级别信息)
            if nt:
                # Matches 匹配逻辑
                r = t[..., 4:6] / anchors[:, None]  # wh ratio 真实边框和先验框的尺度比值
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter [m,7] 仅获取高宽比在anchor_t范围内的 --> 得到每个真实边框对应的先验框尺度信息

                # Offsets 得到每个真实边框对应的grid cell
                gxy = t[:, 2:4]  # grid xy 真实边框的中心点
                gxi = gain[[2, 3]] - gxy  # inverse feature map上的反方向的中心点坐标位置
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # 在一个grid cell中，真实边框中心点 偏左j为True，偏上k为True
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # 在一个grid cell中，真实边框中心点 偏右l为True，偏下m为True
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # [5,m] 针对每个边框，对应的5个方向
                t = t.repeat((5, 1, 1))[j]  # [k,7] 总共有k个anchor box对应匹配GT BOX
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # 分割 (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices 对应grid cell的位置(单元格左上角坐标)

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 中心点相比于单元格左上角坐标的偏移值 --> 作为新边框的中心点，gwh作为边框的宽度高度
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
