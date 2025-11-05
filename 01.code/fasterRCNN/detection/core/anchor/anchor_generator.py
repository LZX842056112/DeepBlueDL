import tensorflow as tf
from detection.utils.misc import calc_img_shapes, calc_batch_padded_shape

class AnchorGenerator:
    def __init__(self, 
                 scales=(32, 64, 128, 256, 512), 
                 ratios=(0.5, 1, 2), 
                 feature_strides=(4, 8, 16, 32, 64)):
        '''
        初始化anchor
        '''
        # scales: 生成的anchor的尺度
        self.scales = scales
        # ratios: anchor的长宽比
        self.ratios = ratios
        # feature_strides: 应为fpn生成了五种特征图，在每一个特征图上移动一个位置相当于原图的大小
        self.feature_strides = feature_strides
     
    def generate_pyramid_anchors(self, img_metas):
        '''
        生成anchor
        
        参数：
            img_metas: [batch_size, 11]，图像的信息，包括原始图像的大小，resize的大小和输入到网络中图像的大小
        
        返回：
            anchors: [num_anchors, (y1, x1, y2, x2)] anchor的坐标，在原图像中的坐标
            valid_flags: [batch_size, num_anchors] 是否为空的标志
        '''
        # 获取输入到网络中图像的大小：[1216, 1216]
        pad_shape = calc_batch_padded_shape(img_metas)
        # 获取图像的每一个特征图的大小：[(304, 304), (152, 152), (76, 76), (38, 38), (19, 19)]
        feature_shapes = [(pad_shape[0] // stride, pad_shape[1] // stride)
                          for stride in self.feature_strides]
        # 生成每一个特征图上anchor的位置信息： [277248, 4], [69312, 4], [17328, 4], [4332, 4], [1083, 4]
        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ]
        # 将所有的anchor串联在一个列表中：[369303, 4]
        anchors = tf.concat(anchors, axis=0)

        # 获取图像非0位置的大小：(800, 1067)
        img_shapes = calc_img_shapes(img_metas)
        # 获取anchor的非零标识
        valid_flags = [
            self._generate_valid_flags(anchors, img_shapes[i])
            for i in range(img_shapes.shape[0])
        ]
        # 堆叠为一个一维向量
        valid_flags = tf.stack(valid_flags, axis=0)
        # 停止梯度计算
        anchors = tf.stop_gradient(anchors)
        valid_flags = tf.stop_gradient(valid_flags)
        # 返回anchor和对应非零标志
        return anchors, valid_flags
    
    def _generate_valid_flags(self, anchors, img_shape):
        '''
        移除padding位置的anchor
        参数：
            anchors: [num_anchors, (y1, x1, y2, x2)] 所有的anchor
            img_shape: Tuple. (height, width, channels) 非0像素点的图像的大小
        返回：
            valid_flags: [num_anchors] 返回非0位置的anchor
        '''
        # 计算所有anchor的中心点坐标：[369300]
        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2
        # 初始化flags为全1数组：[369300]
        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        # 初始化相同大小的全0数组
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)
        # 将anchor中心点在非0区域的置为1，其他置为0
        valid_flags = tf.where(y_center <= img_shape[0], valid_flags, zeros)
        valid_flags = tf.where(x_center <= img_shape[1], valid_flags, zeros)
        # 返回标志结果
        return valid_flags
    
    def _generate_level_anchors(self, level, feature_shape):
        '''生成fpn输出的某一个特征图的anchor

        参数：
            feature_shape: (height, width) 特征图大小

        返回：
            numpy.ndarray [anchors_num, (y1, x1, y2, x2)]：生成的anchor结果
        '''
        # 获取对应的尺度
        scale = self.scales[level]
        # 获取长宽比
        ratios = self.ratios
        # 获取对应步长
        feature_stride = self.feature_strides[level]
        
        # 获取不同长宽比下的scale
        scales, ratios = tf.meshgrid([float(scale)], ratios)
        # 尺度 [32, 32, 32]
        scales = tf.reshape(scales, [-1])
        # 长宽比 [0.5, 1, 2]
        ratios = tf.reshape(ratios, [-1])
        
        # 获取不同宽高比情况下的H和w
        # [45, 32, 22]
        heights = scales / tf.sqrt(ratios)
        # [22, 32, 45]
        widths = scales * tf.sqrt(ratios)

        # 获取生成anchor对应的位置,假设步长为4时的结果： [0, 4, ..., 1216-4]
        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)
        # 类型转换
        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        # 获取在图像中生成anchor的位置
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # 将宽高分别相对于x,y进行广播， 得到宽高和中心点坐标
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # 进行reshape得到anchor的中心点坐标和宽高
        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

        # 拼接成一维向量 [304x304, 3, 4] => [277448, 4]
        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)
        # 返回最终的anchorbox
        return boxes
