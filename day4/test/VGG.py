import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # 分类器部分
        self.classifier = nn.Sequential(
            # 第一个全连接层
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 第二个全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 输出层
            nn.Linear(4096, num_classes),
        )
        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平特征图
        x = torch.flatten(x, 1)
        # 分类
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化卷积层权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    """
    根据配置创建VGG网络的特征提取层
    cfg: 配置列表，每个元素表示该层的输出通道数，'M'表示最大池化层
    batch_norm: 是否使用批归一化
    """
    layers = []
    in_channels = 3  # 输入通道数，RGB图像为3
    for v in cfg:
        if v == 'M':
            # 添加最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 添加卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# VGG16配置
cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
}


def vgg16(pretrained=False, progress=True, **kwargs):
    """
    创建VGG-16模型
    pretrained: 是否加载预训练权重
    progress: 是否显示下载进度
    **kwargs: 其他参数，如num_classes等
    """
    # 使用VGG16配置创建特征提取层
    model = VGG(make_layers(cfg['D']), **kwargs)
    
    # 如果需要预训练权重，可以在这里添加加载代码
    if pretrained:
        # 这里仅作示例，实际使用时应使用torchvision的预训练权重
        print("预训练权重加载功能示例")
        # 例如: state_dict = torch.hub.load_state_dict_from_url(model_urls['vgg16'], progress=progress)
        # model.load_state_dict(state_dict)
    
    return model


# 示例用法
def example_usage():
    # 创建VGG16模型
    model = vgg16(num_classes=1000)
    
    # 打印模型结构
    print("VGG16模型结构:")
    print(model)
    
    # 创建一个随机输入张量 (batch_size=4, channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)
    
    # 前向传播
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")


if __name__ == "__main__":
    example_usage()