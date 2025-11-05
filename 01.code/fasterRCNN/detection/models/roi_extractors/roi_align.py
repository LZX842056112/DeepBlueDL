import tensorflow as tf

from detection.utils.misc import *

class PyramidROIAlign(tf.keras.layers.Layer):

    def __init__(self, pool_shape, **kwargs):
        '''
        在多个特征图上完成ROIPooling

        参数：
            pool_shape: (height, width)指明pooling之后输出的大小
        '''
        super(PyramidROIAlign, self).__init__(**kwargs)

        self.pool_shape = tuple(pool_shape)

    def call(self, inputs, training=True):

        # 获取输入中的roi区域，特征图和图像的元信息
        rois_list, feature_map_list, img_metas = inputs
        # 获取输入图像的大小
        pad_shapes = calc_pad_shapes(img_metas)
        # 图像的尺度：1216*1216
        pad_areas = pad_shapes[:, 0] * pad_shapes[:, 1]
        # 获取图像中ROI的类别data:[2000]
        num_rois_list = [rois.shape.as_list()[0] for rois in rois_list]
        # 获取图像中ROI的索引
        roi_indices = tf.constant(
            [i for i in range(len(rois_list)) for _ in range(rois_list[i].shape.as_list()[0])],
            dtype=tf.int32
        ) #[0.....], shape:[2000]
        # 获取对于每一个ROI的图像大小
        areas = tf.constant(#              range(1)                               range(2000)
            [pad_areas[i] for i in range(pad_areas.shape[0]) for _ in range(num_rois_list[i])],
            dtype=tf.float32
        )#[1216*1216, 1216*1216,...], shape:[2000]

        # ROI
        rois = tf.concat(rois_list, axis=0) # [2000, 4]

        # 获取每一个ROI对应的坐标和宽高
        y1, x1, y2, x2 = tf.split(rois, 4, axis=1) # 4 of [2000, 1]
        h = y2 - y1 # [2000, 1]
        w = x2 - x1 # [2000, 1]
        
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        # 将每一个ROI分配到对应的特征图上
        roi_level = tf.math.log( # [2000]
                    tf.sqrt(tf.squeeze(h * w, 1))
                    / tf.cast((224.0 / tf.sqrt(areas * 1.0)), tf.float32)
                    ) / tf.math.log(2.0)
        roi_level = tf.minimum(5, tf.maximum( # [2000], clamp to [2-5]
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # roi_level will indicates which level of feature to use

        
        # 遍历所有的特征图，进行ROIpooling/ROIAlign
        pooled_rois = []
        roi_to_level = []
        for i, level in enumerate(range(2, 6)): # 2,3,4,5
            # 找到ROI对应的特征图尺度
            ix = tf.where(tf.equal(roi_level, level)) # [1999, 1], means 1999 of 2000 select P2
            # 获取到对应的ROI区域
            level_rois = tf.gather_nd(rois, ix) # boxes to crop, [1999, 4]

            # 获取ROI对应的索引
            level_roi_indices = tf.gather_nd(roi_indices, ix) # [19999], data:[0....0]

            # Keep track of which roi is mapped to which level
            roi_to_level.append(ix)

            # 不进行梯度更新
            level_rois = tf.stop_gradient(level_rois)
            level_roi_indices = tf.stop_gradient(level_roi_indices)

            # 进行ROI_align(在MaskRCNN中介绍)
            pooled_rois.append(tf.image.crop_and_resize(
                feature_map_list[i], level_rois, level_roi_indices, self.pool_shape,
                method="bilinear")) # [1, 304, 304, 256], [1999, 4], [1999], [2]=[7,7]=>[1999,7,7,256]
        # [1999, 7, 7, 256], [], [], [1,7,7,256] => [2000, 7, 7, 256]
        # 将特征拼接在一起
        pooled_rois = tf.concat(pooled_rois, axis=0)

        # Pack roi_to_level mapping into one array and add another
        # column representing the order of pooled rois
        roi_to_level = tf.concat(roi_to_level, axis=0) # [2000, 1], 1999 of P2, and 1 other P
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]), 1) # [2000, 1], 0~1999
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range],
                                 axis=1) # [2000, 2], (P, range)

        # Rearrange pooled features to match the order of the original rois
        # Sort roi_to_level by batch then roi indextf.Tensor([        0    100001    200002 ... 199801997 199901998  20101999], shape=(2000,), dtype=int32)
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape( # k=2000
            roi_to_level)[0]).indices[::-1]# reverse the order
        ix = tf.gather(roi_to_level[:, 1], ix) # [2000]
        pooled_rois = tf.gather(pooled_rois, ix) # [2000, 7, 7, 256]
        # 获取 2000个候选区域 2000 of [7, 7, 256]
        pooled_rois_list = tf.split(pooled_rois, num_rois_list, axis=0)
        return pooled_rois_list
