# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/16 21:42
Create User : 19410
Desc : xxx
"""
from torchvision import models


def t0():
    net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    del net.classifier[-1]
    print(net)


def t1():
    """
    前提条件：修改vgg.py文件内容


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            if "num_classes" not in kwargs:
                kwargs['num_classes'] = len(weights.meta["categories"])
            ## 原始代码
            # _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        ckpt = weights.get_state_dict(progress=progress, check_hash=True)
        if weights.meta["categories"] is not None:
            if len(weights.meta["categories"]) != kwargs['num_classes']:
                del ckpt['classifier.6.weight']
                del ckpt['classifier.6.bias']
        model.load_state_dict(ckpt, strict=False)
        ## 原始代码
        # model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model

    :return:
    """
    net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT, num_classes=17, suppet_change_numclasses=True)
    print(net)


if __name__ == '__main__':
    t1()
