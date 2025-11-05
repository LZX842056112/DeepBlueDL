import tensorflow as tf
from detection.core.bbox import geometry, transforms
from detection.utils.misc import trim_zeros



class AnchorTarget:
    """
    对于每一个anchor：[326393, 4]，创建rpn_target_matchs用于RPN网络的训练
    """
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''
        计算每一个anchor回归和分类的目标值
        
        参数：

            target_means: [4]. box的均值
            target_stds: [4]. box的方差.
            num_rpn_deltas: int. 每一张图片进行网络训练的anchor个数
            positive_fraction: float. 正样本所占的比例
            pos_iou_thr: float. 正样本的阈值
            neg_iou_thr: float. 负样本的阈值
        '''
        # box的均值
        self.target_means = target_means
        # box的方差
        self.target_stds = target_stds
        # 每一张图片进行网络训练的anchor个数
        self.num_rpn_deltas = num_rpn_deltas
        # 正样本所占的比例
        self.positive_fraction = positive_fraction
        # 正样本的阈值
        self.pos_iou_thr = pos_iou_thr
        # 负样本的阈值
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''
        给出anchor和gt_box,计算交并比，确定正负样本， 并计算anchor与gt之前的关系
        参数：
            anchors: [num_anchors, (y1, x1, y2, x2)] 图像中的anchors
            valid_flags: [batch_size, num_anchors] anchor的标识
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] gt_box的位置信息
            gt_class_ids: [batch_size, num_gt_boxes] gt_box的类别

        返回：
            rpn_target_matchs: [batch_size, num_anchors] 确定每一个anchor是正负样本
               正样本： 1 , 负样本： -1 , 非正非负：0
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))]  anchorbox的修正值
        '''
        # 返回结果
        rpn_target_matchs = []
        rpn_target_deltas = []
        # 获取batch中的样本个数
        num_imgs = gt_class_ids.shape[0]
        # 遍历所有图像
        for i in range(num_imgs):
            # 获取当前图像的anchor是正负样本和位置信息
            target_match, target_delta = self._build_single_target(
                anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])
            # 并添加到列表中
            rpn_target_matchs.append(target_match)
            rpn_target_deltas.append(target_delta)
        # 拼接成一个列表
        rpn_target_matchs = tf.stack(rpn_target_matchs)
        rpn_target_deltas = tf.stack(rpn_target_deltas)
        # 停止梯度计算
        rpn_target_matchs = tf.stop_gradient(rpn_target_matchs)
        rpn_target_deltas = tf.stop_gradient(rpn_target_deltas)
        # 返回结果
        return rpn_target_matchs, rpn_target_deltas

    def _build_single_target(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''
        计算每幅图像的目标值
        参数：
            anchors: [num_anchors, (y1, x1, y2, x2)] anchor的位置信息
            valid_flags: [num_anchors] anchor的表示
            gt_class_ids: [num_gt_boxes] 真实值的类别
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)] 真实框的位置
        返回：
            target_matchs: [num_anchors] anchor是正负样本
            target_deltas: [num_rpn_deltas, (dy, dx, log(dh), log(dw))] 
        '''
        # 删除为0的真实框
        gt_boxes, _ = trim_zeros(gt_boxes)
        # 初始化全0数组，存储anchor的分类结果
        target_matchs = tf.zeros(anchors.shape[0], dtype=tf.int32)
        
        # 计算anchor与gt之间的交并比 326393 vs 10 => [326393, 10]
        overlaps = geometry.compute_overlaps(anchors, gt_boxes)
        
        neg_values = tf.constant([0, -1])
        pos_values = tf.constant([0, 1])
        
        # 1.设置负样本
        # 获取每一个anchor与各个GT交并比的最大值及其索引
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1])
        # 选择 IOU < 0.3 的 anchor 为 background，标签为 -1
        target_matchs = tf.where(anchor_iou_max < self.neg_iou_thr, 
                                 -tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # 过滤掉pad区域的anchor
        target_matchs = tf.where(tf.equal(valid_flags, 1),
                                 target_matchs, tf.zeros(anchors.shape[0], dtype=tf.int32))

        # 2、选择 IOU > 0.7 的 anchor 为 foreground，标签为 1
        target_matchs = tf.where(anchor_iou_max >= self.pos_iou_thr, 
                                 tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # 3、为每一GT分配一个anchor：不考虑IOU的大小
        # 选择与每一个GT交并比最大的anchor索引 ：[N_gt_boxes]
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        # 将交并比最大的设置为正样本
        target_matchs = tf.compat.v1.scatter_update(tf.Variable(target_matchs), gt_iou_argmax, 1)
        
        # 采样获取正负样本，主要不要使正样本比例超过一半
        # [N_pos_anchors, 1], [15, 1]
        ids = tf.where(tf.equal(target_matchs, 1))
        # 压缩成一个一维向量 [15]
        ids = tf.squeeze(ids, 1)
        # 计算真实正样本个数与所需样本个数之间的差值
        extra = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.positive_fraction)
        # 若差值大于0，说明有足够的正样本
        if extra > 0:
            # 将多余的正样本的标识置为0
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)
        # 获取负样本
        ids = tf.where(tf.equal(target_matchs, -1)) # [213748, 1]
        ids = tf.squeeze(ids, 1)
        # 获取负样本个数与所需负样本个数之间的差值
        extra = ids.shape.as_list()[0] - (self.num_rpn_deltas -
            tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32)))
        # 若差值大于0，则说明有足够的负样本
        if extra > 0:
            # 将多余的负样本置为0
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)
        # 这时我们就有256个anchor,分别包含正负样本.
        
        # 对于每一个正样本，计算其对应的坐标修正值
        # 获取正样本的索引
        ids = tf.where(tf.equal(target_matchs, 1)) # [15]
        # 获取正样本的anchor
        a = tf.gather_nd(anchors, ids)
        # 获取anchor对应的gt的index
        anchor_idx = tf.gather_nd(anchor_iou_argmax, ids)
        # 获取gt
        gt = tf.gather(gt_boxes, anchor_idx)
        # 计算anchor到gt的修正坐标。
        target_deltas = transforms.bbox2delta(
            a, gt, self.target_means, self.target_stds)
        # 获取负样本的个数
        padding = tf.maximum(self.num_rpn_deltas - tf.shape(target_deltas)[0], 0)
        # 目标值，正样本的目标值是偏移，负样本的目标值是0
        target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)])

        return target_matchs, target_deltas