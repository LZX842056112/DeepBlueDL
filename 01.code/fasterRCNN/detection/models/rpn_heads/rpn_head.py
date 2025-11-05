import  tensorflow as tf
from    tensorflow.keras import layers

from detection.core.bbox import transforms
from detection.utils.misc import *

from detection.core.anchor import anchor_generator, anchor_target
from detection.core.loss import losses

class RPNHead(tf.keras.Model):
    """
    完成RPN网络中的相关操作
    """

    def __init__(self, 
                 anchor_scales=(32, 64, 128, 256, 512), 
                 anchor_ratios=(0.5, 1, 2), 
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 proposal_count=2000, 
                 nms_threshold=0.7, 
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 **kwags):
        '''
        RPN网络结构，如下所示：

                                      / - rpn_cls 分类(1x1 conv)
        输入 - rpn_conv 卷积(3x3 conv) -
                                      \ - rpn_reg 回归(1x1 conv)

        参数
            anchor_scales: anchorbox的面积，相对于原图像像素的
            anchor_ratios: anchorbox的长宽比
            anchor_feature_strides: 生成anchor的步长，相对于原图像素的
            proposal_count:RPN最后生成的候选区域的个数，经过非极大值抑制
            nms_threshold: 对RPN生成的候选区域进行NMS的参数阈值
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: 与GT的IOU大于该值的anchor为正例
            neg_iou_thr: 与GT的IOU小于该值的anchor为负例
        '''
        super(RPNHead, self).__init__(**kwags)
        # 参数初始化
        # RPN最后生成的候选区域的个数，经过非极大值抑制
        self.proposal_count = proposal_count
        # 对RPN生成的候选区域进行NMS的参数阈值
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds
        # 调用anchor生成器生成对应的anchor
        self.generator = anchor_generator.AnchorGenerator(
            scales=anchor_scales, 
            ratios=anchor_ratios, 
            feature_strides=anchor_feature_strides)
        # 将anchor划分为正负样本
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=target_means, 
            target_stds=target_stds,
            num_rpn_deltas=num_rpn_deltas,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)
        # 设置RPN网络的分类和回归损失
        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss
        
        
        # 3*3卷积
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal', 
                                             name='rpn_conv_shared')
        # 1*1卷积 分类 每一个anchor分为2类
        self.rpn_class_raw = layers.Conv2D(len(anchor_ratios) * 2, (1, 1),
                                           kernel_initializer='he_normal', 
                                           name='rpn_class_raw')
        # 1*1卷积 回归 每一个anchor的回归结果
        self.rpn_delta_pred = layers.Conv2D(len(anchor_ratios) * 4, (1, 1),
                                           kernel_initializer='he_normal', 
                                           name='rpn_bbox_pred')
        
    def call(self, inputs, training=True):
        '''
        定义前向传播过程
        参数：
            inputs: [batch_size, feat_map_height, feat_map_width, channels] 
                FPN输出的一个特征图
        
        返回：
            rpn_class_logits: [batch_size, num_anchors, 2] 分类结果，以logits表示
            rpn_probs: [batch_size, num_anchors, 2] 分类结果，经softmax之后的概率表示形式
            rpn_deltas: [batch_size, num_anchors, 4] 回归结果，anchor的位置信息
        '''
        # 输出结果
        layer_outputs = []
        # 遍历输入中的每一特征图
        for feat in inputs:
            """
            (1, 304, 304, 256)
            (1, 152, 152, 256)
            (1, 76, 76, 256)
            (1, 38, 38, 256)
            (1, 19, 19, 256)
            rpn_class_raw: (1, 304, 304, 6)
            rpn_class_logits: (1, 277248, 2)
            rpn_delta_pred: (1, 304, 304, 12)
            rpn_deltas: (1, 277248, 4)
            rpn_class_raw: (1, 152, 152, 6)
            rpn_class_logits: (1, 69312, 2)
            rpn_delta_pred: (1, 152, 152, 12)
            rpn_deltas: (1, 69312, 4)
            rpn_class_raw: (1, 76, 76, 6)
            rpn_class_logits: (1, 17328, 2)
            rpn_delta_pred: (1, 76, 76, 12)
            rpn_deltas: (1, 17328, 4)
            rpn_class_raw: (1, 38, 38, 6)
            rpn_class_logits: (1, 4332, 2)
            rpn_delta_pred: (1, 38, 38, 12)
            rpn_deltas: (1, 4332, 4)
            rpn_class_raw: (1, 19, 19, 6)
            rpn_class_logits: (1, 1083, 2)
            rpn_delta_pred: (1, 19, 19, 12)
            rpn_deltas: (1, 1083, 4)

            """
            # 3*3 卷积，假设特征图大小为：(1, 304, 304, 256)
            shared = self.rpn_conv_shared(feat)
            # 激活：(1, 304, 304, 256)
            shared = tf.nn.relu(shared)

            # 分类过程
            # 1*1卷积：输出大小为(1, 304, 304, 6)
            x = self.rpn_class_raw(shared)
            # reshape:(1, 277248, 2)
            rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
            # softmax进行分类：(1, 277248, 2)，一共有277248个anchor，每个anchor有2个分类结果
            rpn_probs = tf.nn.softmax(rpn_class_logits)

            # 回归过程
            # 1*1 卷积，输出大小为(1, 304, 304, 12)
            x = self.rpn_delta_pred(shared)
            # reshape:(1, 277248, 4),一共有277248个anchor，每个anchor有4个位置信息
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])

            # 将网络的分类和输出结果存放在layer_outputs
            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])
            # 每一次迭代输出结果的大小为：
            """
            (1, 277248, 2) (1, 277248, 2) (1, 277248, 4)
            (1, 69312, 2) (1, 69312, 2) (1, 69312, 4)
            (1, 17328, 2) (1, 17328, 2) (1, 17328, 4)
            (1, 4332, 2) (1, 4332, 2) (1, 4332, 4)
            (1, 1083, 2) (1, 1083, 2) (1, 1083, 4)

            """

        # 将输出结果转换为列表
        outputs = list(zip(*layer_outputs))
        # 遍历输出，将不同特征图中同一类别的输出结果串联在一起
        outputs = [tf.concat(list(o), axis=1) for o in outputs]
        # 获取每一种输出：4个特征图的输出大小为：(1, 369303, 2) (1, 369303, 2) (1, 369303, 4)
        rpn_class_logits, rpn_probs, rpn_deltas = outputs

        # 返回输出结果
        return rpn_class_logits, rpn_probs, rpn_deltas

    def loss(self, rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas):
        """
        损失函数
        :param rpn_class_logits: [N, 2] rpn分类输出结果
        :param rpn_deltas: [N, 4]  rpn回归的输出结果
        :param gt_boxes:  [GT_N] 真实值
        :param gt_class_ids:  [GT_N] 真实类别
        :param img_metas: [11] 图像的元信息
        :return:
        """
        # 根据图像元信息产生anchor，并利用valid_flags来标识图像区域的anchor
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)

        # 指定生成的anchor对应的真实值：包括类别及位置
        rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(
            anchors, valid_flags, gt_boxes, gt_class_ids)
        # 计算分类损失
        rpn_class_loss = self.rpn_class_loss(
            rpn_target_matchs, rpn_class_logits)
        # 计算回归损失
        rpn_bbox_loss = self.rpn_bbox_loss(
            rpn_target_deltas, rpn_target_matchs, rpn_deltas)
        # 返回损失结果
        return rpn_class_loss, rpn_bbox_loss
    
    def get_proposals(self, 
                      rpn_probs, 
                      rpn_deltas, 
                      img_metas, 
                      with_probs=False):
        '''
        计算候选区域proposals
        
        参数：
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]  rpn的分类结果
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))] rpn的回归结果
            img_metas: [batch_size, 11] 图像元信息
            with_probs: 是否返回分类结果
        
        返回：
            proposals_list: 候选区域的列表
                若with_probs = False，则返回：[num_proposals, (y1, x1, y2, x2)]
                若with_probs = True，则返回：[num_proposals, (y1, x1, y2, x2, score)]
                在这里num_proposals不会大于proposal_count
        '''
        # 根据图像元信息产生anchor，并利用valid_flags来标识图像区域的anchor
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)

        # 获取anchor为目标的概率 [1, 369303]
        rpn_probs = rpn_probs[:, :, 1]
        # 获取输入到网络中图像的大小：[[1216, 1216]]
        pad_shapes = calc_pad_shapes(img_metas)
        # 获取最终的候选区域的类别
        proposals_list = [
            self._get_proposals_single(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], pad_shapes[i], with_probs)
            for i in range(img_metas.shape[0])
        ]
        # 返回结果
        return proposals_list
    
    def _get_proposals_single(self, 
                              rpn_probs, 
                              rpn_deltas, 
                              anchors, 
                              valid_flags, 
                              img_shape, 
                              with_probs):
        '''
        计算候选区域结果
        
        参数：
            rpn_probs: [num_anchors] anchor是目标的概率值
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))] 回归得到的位置信息，对anchor进行修正
            anchors: [num_anchors, (y1, x1, y2, x2)] anchor的位置
            valid_flags: [num_anchors] anchor属于图像位置的标记信息
            img_shape: np.ndarray. [2]. (img_height, img_width) 图像的大小
            with_probs: bool. 是否输出分类结果
        
        返回
            proposals:  返回候选区域的列表
            若with_probs = False，则返回：[num_proposals, (y1, x1, y2, x2)]
                若with_probs = True，则返回：[num_proposals, (y1, x1, y2, x2, score)]
                在这里num_proposals不会大于proposal_count
        '''
        # 图像的高宽
        H, W = img_shape
        
        # 将anchor的标记信息转换为布尔型, int => bool
        valid_flags = tf.cast(valid_flags, tf.bool)
        # 将无用的anchor过滤 ，并对分类和回归结果进行处理[369303] => [215169], respectively
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        anchors = tf.boolean_mask(anchors, valid_flags)

        # 至多6000个结果会进行后续操作 min(6000, 215169) => 6000
        pre_nms_limit = min(6000, anchors.shape[0])
        # 获取至多6000个分类概率最高的anchor的索引
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
        # 根据得到的索引值获取对应的分类，回归和anchor [215169] => [6000]
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)
        
        # 利用回归得到的结果对anchor进行修正,  [6000, 4]
        proposals = transforms.delta2bbox(anchors, rpn_deltas, 
                                          self.target_means, self.target_stds)
        # 若修正后的结果超出图像范围则进行裁剪, [6000, 4]
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = transforms.bbox_clip(proposals, window)
        
        # 对坐标值进行归一化, (y1, x1, y2, x2)
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)
        
        # 进行NMS，获取最终大概2000个候选区域: [2000]
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        proposals = tf.gather(proposals, indices) # [2000, 4]
        # 若要返回分类结果，则获取对应的分类值进行返回
        if with_probs:
            proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
            proposals = tf.concat([proposals, proposal_probs], axis=1)
        # 返回候选区域
        return proposals
        
        