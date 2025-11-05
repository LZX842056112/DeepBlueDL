import numpy as np
import tensorflow as tf

from detection.core.bbox import geometry, transforms
from detection.utils.misc import *

class ProposalTarget:
    # 设置候选区域的正负样本
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5):
        '''
        设置候选区域的正负样本
        
        参数：
            target_means: [4]. 候选区域bbox的均值
            target_stds: [4]. 候选区域bbox的标准差
            num_rcnn_deltas: int. 送入网络中进行预测的bbox的个数

        '''
        # 候选区域bbox的均值
        self.target_means = target_means
        # 候选区域bbox的标准差
        self.target_stds = target_stds
        # 送入网络中进行预测的bbox的个数
        self.num_rcnn_deltas = num_rcnn_deltas
        # 正样本的比例
        self.positive_fraction = positive_fraction
        # 超出该阈值的为正样本
        self.pos_iou_thr = pos_iou_thr
        # 低于该阈值的为负样本
        self.neg_iou_thr = neg_iou_thr
            
    def build_targets(self, proposals_list, gt_boxes, gt_class_ids, img_metas):
        '''
        生成网络的候选区域，并从候选区域中采样送入后续网络中，生成其对应的类别和bbox的修正值
        参数：
            proposals_list:  [num_proposals, (y1, x1, y2, x2)] 候选区域的列表，归一化坐标
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] 真实值，图像坐标
            gt_class_ids: [batch_size, num_gt_boxes] 真实值，类别
            img_metas: [batch_size, 11] 图像元信息
            
        返回：
            rois_list: [num_rois, (y1, x1, y2, x2)] 采样后获取区域列表，归一化坐标
            rcnn_target_matchs_list: list of [num_rois]. 类别信息
            rcnn_target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))]. 回归目标值

        '''
        # 图像大小：# [[1216, 1216]]
        pad_shapes = calc_pad_shapes(img_metas)
        # 返回结果存放在列表中
        rois_list = []
        rcnn_target_matchs_list = []
        rcnn_target_deltas_list = []
        # 遍历batch中的图像
        for i in range(img_metas.shape[0]):
            # 获取roi区域，对应的类别和回归结果
            rois, target_matchs, target_deltas = self._build_single_target(
                proposals_list[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i])
            # [192, 4] 包括正负样本
            rois_list.append(rois)
            # 正样本对应的类别，负样本用0填充
            rcnn_target_matchs_list.append(target_matchs)
            # 正样本对应的坐标，负样本用0填充
            rcnn_target_deltas_list.append(target_deltas)
        # 返回结果
        return rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list
    
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape):
        '''
        生成一幅图像中的正负样本
        参数：
            proposals: [num_proposals, (y1, x1, y2, x2)] rpn网络生成的候选区域：归一化坐标
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)] 图像中真实值，bbox的坐标值，图像坐标
            gt_class_ids: [num_gt_boxes] 图像中的gt对应的类别
            img_shape: np.ndarray. [2]. (img_height, img_width) 图像的大小
            
        返回：
            rois: [num_rois, (y1, x1, y2, x2)] 候选区域的归一化坐标
            target_matchs: [num_positive_rois] 采样后候选区域的类别
            target_deltas: [num_positive_rois, (dy, dx, log(dh), log(dw))] 采样后候选区域的目标值
        '''
        # 图像的大小
        H, W = img_shape # 1216, 1216
        # 移除0值 [7, 4]
        gt_boxes, non_zeros = trim_zeros(gt_boxes)
        # 获取GT对应的类别
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros) # [7]
        # 归一化 (y1, x1, y2, x2) => 0~1
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)
        # 计算候选区域和真实框之间的交并比：[2k, 4] with [7, 4] => [2k, 7]
        overlaps = geometry.compute_overlaps(proposals, gt_boxes)
        # 获取每一个候选区域最相似的gtbox的索引[2000]
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        # 获取每一个候选区域与最相似的gtbox的交并比[2000]
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 获取正样本的索引[2000]=>[48, 1] =>[48]
        positive_roi_bool = (roi_iou_max >= self.pos_iou_thr)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 获取负样本的索引
        negative_indices = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]

        # 对获取的ROI区域进行下采样
        # 需要的正样本个数，通过比例计算
        positive_count = int(self.num_rcnn_deltas * self.positive_fraction)
        # 将正样本打乱，进行截取
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        # 正样本的个数
        positive_count = tf.shape(positive_indices)[0]
        
        # 负样本，保证正样本的比例
        r = 1.0 / self.positive_fraction
        # 计算样本总数并减去正样本个数，即为负样本个数
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        # 获取负样本
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        
        # 选取正负样本的候选区域
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)
        
        # 为选取的候选区域分配目标值，获取正样本与GT的交并比
        positive_overlaps = tf.gather(overlaps, positive_indices)
        # 获取与每一个候选区域最相似的GT
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        # 将GT的坐标和类别分配给对应的候选区域
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        target_matchs = tf.gather(gt_class_ids, roi_gt_box_assignment)
        # 将坐标转换为修正值
        target_deltas = transforms.bbox2delta(positive_rois, roi_gt_boxes, self.target_means, self.target_stds)
        # 将正负样本拼接在一起
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        # 获取负样本的数量
        N = tf.shape(negative_rois)[0]
        # 将负样本类别设为0
        target_matchs = tf.pad(target_matchs, [(0, N)])
        # 停止梯度更新
        target_matchs = tf.stop_gradient(target_matchs)
        target_deltas = tf.stop_gradient(target_deltas)
        # 返回结果
        return rois, target_matchs, target_deltas