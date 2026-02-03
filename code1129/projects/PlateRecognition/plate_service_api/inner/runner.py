# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/29 14:14
Create User : 19410
Desc : xxx
"""

import logging

from .plate_detect_model import PlateDetectionModel
from .plate_ocr import OCRRecognition


class PlateService(object):
    def __init__(self):
        super(PlateService, self).__init__()
        # 车牌区域检测模型/方法/调用的接口
        self.plate_detect_model = PlateDetectionModel()
        # 车牌号码的提取模型/方法/接口
        self.plate_ocr = OCRRecognition()

    def predict(self, image):
        """
        基于传入的image图像，返回车牌号码
        """
        try:
            # 1. 参数的检查
            if image is None:
                return {'code': 1, 'msg': '图像对象为空'}

            # 2. 从图像中提取车牌区域
            img = self.plate_detect_model.predict(image)
            if img is None:
                return {'code': 1, 'msg': '未检测出车牌区域.'}

            # 3. 检测车牌号码
            result = self.plate_ocr.predict(img)
            if result is None or 'text' not in result:
                return {'code': 1, 'msg': '车牌号码提取失败'}
            text = result['text']
            if len(text) == 0:
                return {'code': 1, 'msg': '未成功提取车牌号码，车牌号码不存在'}
            else:
                return {'code': 0, 'msg': '成功提取.', 'data': text}
        except Exception as e:
            logging.error("车牌号码提取异常.", exc_info=e)
            return {
                'code': 1,
                'msg': f'车牌号码提取异常:{e}'
            }
