'''
FasterRCNN的backbone的实现
'''
import  tensorflow as tf
from    tensorflow.keras import layers

class _Bottleneck(tf.keras.Model):
    """
    瓶颈模块的实现
    """
    def __init__(self, filters, block, 
                 downsampling=False, stride=1, **kwargs):
        super(_Bottleneck, self).__init__(**kwargs)
        # 获取三个卷积的卷积核数量
        filters1, filters2, filters3 = filters
        # 卷积层命名方式
        conv_name_base = 'res' + block + '_branch'
        # BN层命名方式
        bn_name_base   = 'bn'  + block + '_branch'
        # 是否进行下采样
        self.downsampling = downsampling
        # 卷积步长
        self.stride = stride
        # 瓶颈模块输出的通道数
        self.out_channel = filters3
        # 1*1 卷积
        self.conv2a = layers.Conv2D(filters1, (1, 1), strides=(stride, stride),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2a')
        # BN层
        self.bn2a = layers.BatchNormalization(name=bn_name_base + '2a')
        # 3*3 卷积
        self.conv2b = layers.Conv2D(filters2, (3, 3), padding='same',
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2b')
        # BN层
        self.bn2b = layers.BatchNormalization(name=bn_name_base + '2b')
        # 1*1卷积
        self.conv2c = layers.Conv2D(filters3, (1, 1),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2c')
        # BN层
        self.bn2c = layers.BatchNormalization(name=bn_name_base + '2c')
        # 下采样
        if self.downsampling:
            # 在短连接处进行下采样
            self.conv_shortcut = layers.Conv2D(filters3, (1, 1), strides=(stride, stride),
                                               kernel_initializer='he_normal',
                                               name=conv_name_base + '1')
            # BN层
            self.bn_shortcut = layers.BatchNormalization(name=bn_name_base + '1')
    
    def call(self, inputs, training=False):
        """
        定义前向传播过程
        :param inputs:
        :param training:
        :return:
        """
        # 第一组卷积+BN+Relu
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        # 第二组卷积+BN+Relu
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        # 第三组卷积+BN
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        # 短连接
        if self.downsampling:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut, training=training)
        else:
            shortcut = inputs
        # 相加求和
        x += shortcut
        # 激活
        x = tf.nn.relu(x)
        # 最终输出
        return x
    
    def compute_output_shape(self, input_shape):
        """
        计算输出特征图的大小，未使用
        :param input_shape:
        :return:
        """
        # 获取输入的大小
        shape = tf.TensorShape(input_shape).as_list()
        # 获取输出的大小
        shape[1] = shape[1] // self.stride
        shape[2] = shape[2] // self.stride
        shape[-1] = self.out_channel
        return tf.TensorShape(shape)        
        

class ResNet(tf.keras.Model):
    "构建50或101层的resnet网络"
    def __init__(self, depth, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        # 若深度不是50或101报错
        if depth not in [50, 101]:
            raise AssertionError('depth must be 50 or 101.')
        self.depth = depth
        # padding
        self.padding = layers.ZeroPadding2D((3, 3))
        # 输入的卷积
        self.conv1 = layers.Conv2D(64, (7, 7),
                                   strides=(2, 2),
                                   kernel_initializer='he_normal',
                                   name='conv1')
        # BN层
        self.bn_conv1 = layers.BatchNormalization(name='bn_conv1')
        # maxpooling
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        # 第一组瓶颈模块：首个进行下采样
        self.res2a = _Bottleneck([64, 64, 256], block='2a',
                                 downsampling=True, stride=1)
        self.res2b = _Bottleneck([64, 64, 256], block='2b')
        self.res2c = _Bottleneck([64, 64, 256], block='2c')
        # 第二组瓶颈模块：首个进行下采样
        self.res3a = _Bottleneck([128, 128, 512], block='3a', 
                                 downsampling=True, stride=2)
        self.res3b = _Bottleneck([128, 128, 512], block='3b')
        self.res3c = _Bottleneck([128, 128, 512], block='3c')
        self.res3d = _Bottleneck([128, 128, 512], block='3d')
        # 第三组瓶颈模块：首个进行下采样
        self.res4a = _Bottleneck([256, 256, 1024], block='4a', 
                                 downsampling=True, stride=2)
        self.res4b = _Bottleneck([256, 256, 1024], block='4b')
        self.res4c = _Bottleneck([256, 256, 1024], block='4c')
        self.res4d = _Bottleneck([256, 256, 1024], block='4d')
        self.res4e = _Bottleneck([256, 256, 1024], block='4e')
        self.res4f = _Bottleneck([256, 256, 1024], block='4f')
        # 若深度为101还需进行瓶颈模块的串联
        if self.depth == 101:
            self.res4g = _Bottleneck([256, 256, 1024], block='4g')
            self.res4h = _Bottleneck([256, 256, 1024], block='4h')
            self.res4i = _Bottleneck([256, 256, 1024], block='4i')
            self.res4j = _Bottleneck([256, 256, 1024], block='4j')
            self.res4k = _Bottleneck([256, 256, 1024], block='4k')
            self.res4l = _Bottleneck([256, 256, 1024], block='4l')
            self.res4m = _Bottleneck([256, 256, 1024], block='4m')
            self.res4n = _Bottleneck([256, 256, 1024], block='4n')
            self.res4o = _Bottleneck([256, 256, 1024], block='4o')
            self.res4p = _Bottleneck([256, 256, 1024], block='4p')
            self.res4q = _Bottleneck([256, 256, 1024], block='4q')
            self.res4r = _Bottleneck([256, 256, 1024], block='4r')
            self.res4s = _Bottleneck([256, 256, 1024], block='4s')
            self.res4t = _Bottleneck([256, 256, 1024], block='4t')
            self.res4u = _Bottleneck([256, 256, 1024], block='4u')
            self.res4v = _Bottleneck([256, 256, 1024], block='4v')
            self.res4w = _Bottleneck([256, 256, 1024], block='4w') 
        # 第四组瓶颈模块：首个进行下采样
        self.res5a = _Bottleneck([512, 512, 2048], block='5a', 
                                 downsampling=True, stride=2)
        self.res5b = _Bottleneck([512, 512, 2048], block='5b')
        self.res5c = _Bottleneck([512, 512, 2048], block='5c')
        # 输出通道数：C2,C3,C4,C5的输出通道数
        self.out_channel = (256, 512, 1024, 2048)
    
    def call(self, inputs, training=True):
        "定义前向传播过程，每组瓶颈模块均输出结果"
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        # 第1组瓶颈模块：输出c2
        x = self.res2a(x, training=training)
        x = self.res2b(x, training=training)
        C2 = x = self.res2c(x, training=training)
        # 第2组瓶颈模块:输出c3
        x = self.res3a(x, training=training)
        x = self.res3b(x, training=training)
        x = self.res3c(x, training=training)
        C3 = x = self.res3d(x, training=training)
        # 第3组瓶颈模块:输出c4
        x = self.res4a(x, training=training)
        x = self.res4b(x, training=training)
        x = self.res4c(x, training=training)
        x = self.res4d(x, training=training)
        x = self.res4e(x, training=training)
        x = self.res4f(x, training=training)
        if self.depth == 101:
            x = self.res4g(x, training=training)
            x = self.res4h(x, training=training)
            x = self.res4i(x, training=training)
            x = self.res4j(x, training=training)
            x = self.res4k(x, training=training)
            x = self.res4l(x, training=training)
            x = self.res4m(x, training=training)
            x = self.res4n(x, training=training)
            x = self.res4o(x, training=training)
            x = self.res4p(x, training=training)
            x = self.res4q(x, training=training)
            x = self.res4r(x, training=training)
            x = self.res4s(x, training=training)
            x = self.res4t(x, training=training)
            x = self.res4u(x, training=training)
            x = self.res4v(x, training=training)
            x = self.res4w(x, training=training) 
        C4 = x
        # 第4组瓶颈模块:输出c5
        x = self.res5a(x, training=training)
        x = self.res5b(x, training=training)
        C5 = x = self.res5c(x, training=training)
        # 返回所有的输出送入到fpn中
        return (C2, C3, C4, C5)
    
    def compute_output_shape(self, input_shape):
        '计算输出特征图的大小，未使用'
        # 获取输入的大熊啊
        shape = tf.TensorShape(input_shape).as_list()
        batch, H, W, C = shape
        # 获取输出的大小
        C2_shape = tf.TensorShape([batch, H //  4, W //  4, self.out_channel[0]])
        C3_shape = tf.TensorShape([batch, H //  8, W //  8, self.out_channel[1]])
        C4_shape = tf.TensorShape([batch, H // 16, W // 16, self.out_channel[2]])
        C5_shape = tf.TensorShape([batch, H // 32, W // 32, self.out_channel[3]])
        # 返回输出的形状
        return (C2_shape, C3_shape, C4_shape, C5_shape)