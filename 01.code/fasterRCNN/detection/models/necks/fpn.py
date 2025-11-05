'''
构建FPN模块
'''

import tensorflow as tf
from tensorflow.keras import layers


class FPN(tf.keras.Model):

    def __init__(self, out_channels=256, **kwargs):
        '''
        构建FPN模块：
        out_channels:是输出特征图的通道数
        '''
        super(FPN, self).__init__(**kwargs)
        # 输出通道数
        self.out_channels = out_channels
        # 使用1*1卷积对每个输入的特征图进行通道数调整
        self.fpn_c2p2 = layers.Conv2D(out_channels, (1, 1),
                                      kernel_initializer='he_normal', name='fpn_c2p2')
        self.fpn_c3p3 = layers.Conv2D(out_channels, (1, 1),
                                      kernel_initializer='he_normal', name='fpn_c3p3')
        self.fpn_c4p4 = layers.Conv2D(out_channels, (1, 1),
                                      kernel_initializer='he_normal', name='fpn_c4p4')
        self.fpn_c5p5 = layers.Conv2D(out_channels, (1, 1),
                                      kernel_initializer='he_normal', name='fpn_c5p5')
        # 对深层的特征图进行上采样，使其与前一层的大小相同
        self.fpn_p3upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p3upsampled')
        self.fpn_p4upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p4upsampled')
        self.fpn_p5upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')
        # 3*3卷积，作用于融合后的特征图中得到最终的结果
        self.fpn_p2 = layers.Conv2D(out_channels, (3, 3), padding='SAME',
                                    kernel_initializer='he_normal', name='fpn_p2')
        self.fpn_p3 = layers.Conv2D(out_channels, (3, 3), padding='SAME',
                                    kernel_initializer='he_normal', name='fpn_p3')
        self.fpn_p4 = layers.Conv2D(out_channels, (3, 3), padding='SAME',
                                    kernel_initializer='he_normal', name='fpn_p4')
        self.fpn_p5 = layers.Conv2D(out_channels, (3, 3), padding='SAME',
                                    kernel_initializer='he_normal', name='fpn_p5')
        # 对上一层的特征图进行下采样得到结果
        self.fpn_p6 = layers.MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')

    def call(self, inputs, training=True):
        # 定义前向传播过程
        # 获取从resnet中得到的4个特征图
        C2, C3, C4, C5 = inputs
        # 对这些特征图进行1*1卷积和上采样后进行融合
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + self.fpn_p5upsampled(P5)
        P3 = self.fpn_c3p3(C3) + self.fpn_p4upsampled(P4)
        P2 = self.fpn_c2p2(C2) + self.fpn_p3upsampled(P3)
        # 对融合后的特征图进行3*3卷积，得到最终的结果
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        # 对p5进行下采样得到p6特征图
        P6 = self.fpn_p6(P5)
        # 返回最终的结果
        return [P2, P3, P4, P5, P6]

    def compute_output_shape(self, input_shape):
        "计算输出特征图的大小，未使用"
        # 获取输入特征图的大小
        C2_shape, C3_shape, C4_shape, C5_shape = input_shape
        # 转换为类别
        C2_shape, C3_shape, C4_shape, C5_shape = \
            C2_shape.as_list(), C3_shape.as_list(), C4_shape.as_list(), C5_shape.as_list()
        # 通过c5计算得出c6的大小
        C6_shape = [C5_shape[0], (C5_shape[1] + 1) // 2, (C5_shape[2] + 1) // 2, self.out_channels]
        # 输出通道数
        C2_shape[-1] = self.out_channels
        C3_shape[-1] = self.out_channels
        C4_shape[-1] = self.out_channels
        C5_shape[-1] = self.out_channels
        # 输出特征图的大小
        return [tf.TensorShape(C2_shape),
                tf.TensorShape(C3_shape),
                tf.TensorShape(C4_shape),
                tf.TensorShape(C5_shape),
                tf.TensorShape(C6_shape)]


if __name__ == '__main__':
    C2 = tf.random.normal((2, 256, 256, 256))
    C3 = tf.random.normal((2, 128, 128, 512))
    C4 = tf.random.normal((2, 64, 64, 1024))
    C5 = tf.random.normal((2, 32, 32, 2048))

    fpn = FPN()

    P2, P3, P4, P5, P6 = fpn([C2, C3, C4, C5])

    print('P2 shape:', P2.shape.as_list())
    print('P3 shape:', P3.shape.as_list())
    print('P4 shape:', P4.shape.as_list())
    print('P5 shape:', P5.shape.as_list())
    print('P6 shape:', P6.shape.as_list())
