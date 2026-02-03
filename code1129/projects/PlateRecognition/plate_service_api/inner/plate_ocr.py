# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/29 14:16
Create User : 19410
Desc : xxx
"""


class OCRRecognition(object):
    def __init__(self):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model = 'iic/cv_convnextTiny_ocr-recognition-licenseplate_damo'
        self.ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model=model
        )

    def predict(self, img):
        return self.ocr_recognition(img)

    def __call__(self, img, **kwargs):
        return self.predict(img)
