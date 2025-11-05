import tensorflow as tf
from    tensorflow import keras


def smooth_l1_loss(y_true, y_pred):
    '''Implements Smooth-L1 loss.
    
    Args
    ---
        y_true and y_pred are typically: [N, 4], but could be any shape.
    '''
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss



def rpn_class_loss(target_matchs, rpn_class_logits):
    '''RPN分类损失
    参数：
        target_matchs: [batch_size, num_anchors]. anchor的标记信息. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch_size, num_anchors, 2]. RPN的分类结果 FG/BG.
    '''

    # 获取anchor的分类标记信息. 将 -1/+1 转换为 0/1 值
    anchor_class = tf.cast(tf.equal(target_matchs, 1), tf.int32)
    # 正负样本对损失都有贡献，获取正负样本的索引
    indices = tf.where(tf.not_equal(target_matchs, 0))
    # 获取正负样本对应的预测值
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    # 获取正负样本对应的真实累呗
    anchor_class = tf.gather_nd(anchor_class, indices)
    # 获取类别个数
    num_classes = rpn_class_logits.shape[-1]
    # 计算交叉熵损失结果
    loss = keras.losses.categorical_crossentropy(tf.one_hot(anchor_class, depth=num_classes),
                                                 rpn_class_logits, from_logits=True)

    # 求平均
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    # 返回loss值
    return loss


def rpn_bbox_loss(target_deltas, target_matchs, rpn_deltas):
    '''
    rpn损失的回归结果
    参数：
        target_deltas: [batch, num_rpn_deltas, (dy, dx, log(dh), log(dw))].
        target_matchs: [batch, anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_deltas: [batch, anchors, (dy, dx, log(dh), log(dw))]
    '''
    def batch_pack(x, counts, num_rows):
        # 获取指定的位置的值
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)
    
    # 只有正样本计算损失，获取正样本的索引
    indices = tf.where(tf.equal(target_matchs, 1))
    # 获取正样本对应的预测值
    rpn_deltas = tf.gather_nd(rpn_deltas, indices)

    # 获取正样本的个数
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32), axis=1)
    # 获取正样本对应的目标值
    target_deltas = batch_pack(target_deltas, batch_counts,
                              target_deltas.shape.as_list()[0])
    # 计算smoothL1损失
    loss = smooth_l1_loss(target_deltas, rpn_deltas)
    # 计算均值
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    # 返回损失
    return loss





def rcnn_class_loss(target_matchs_list, rcnn_class_logits_list):
    '''FastRCNN的分类损失
    
    参数：
        target_matchs_list:  [num_rois]. 正样本的候选区域
        rcnn_class_logits_list: list of [num_rois, num_classes] 分类结果
    '''
    # 增加背景类的类别
    class_ids = tf.concat(target_matchs_list, 0)
    # 背景类的分数
    class_logits = tf.concat(rcnn_class_logits_list, 0)
    # 类型转换
    class_ids = tf.cast(class_ids, 'int64')
    
    # 获取类别总数
    num_classes = class_logits.shape[-1]
    # 计算交叉熵损失函数
    loss = keras.losses.categorical_crossentropy(tf.one_hot(class_ids, depth=num_classes),
                                                 class_logits, from_logits=True)

    # 求平均：大于0返回结果，其他返回0
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    return loss


def rcnn_bbox_loss(target_deltas_list, target_matchs_list, rcnn_deltas_list):
    '''FastRCNN的回归损失
    
    参数：
        target_deltas_list: [num_positive_rois, (dy, dx, log(dh), log(dw))] 正样本对应的真实值
        target_matchs_list: list of [num_rois]. 正样本对应的类别
        rcnn_deltas_list: list of [num_rois, num_classes, (dy, dx, log(dh), log(dw))] 网络返回的结果
    '''
    # 其他结果为0
    target_deltas = tf.concat(target_deltas_list, 0)
    target_class_ids = tf.concat(target_matchs_list, 0)
    rcnn_deltas = tf.concat(rcnn_deltas_list, 0)

    # 只有正样本参与损失计算，并且只有类别预测正确才获取其索引
    # 获取非背景类的结果
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    # 将类别和回归结果合并在一起
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    # 获取索引
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    
    # 获取正样本预测结果
    rcnn_deltas = tf.gather_nd(rcnn_deltas, indices)

    # 计算Smooth-L1损失
    loss = smooth_l1_loss(target_deltas, rcnn_deltas)
    # 平均：大于0返回结果，其他返回0
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss
