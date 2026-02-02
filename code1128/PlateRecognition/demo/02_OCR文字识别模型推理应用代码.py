# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/28 21:06
Create User : 19410
Desc : 基于给定的图像输出对应的文字内容 --> 可以认为输入的图像中一定有文字/单词

https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-licenseplate_damo/summary
https://arxiv.org/pdf/2010.11929
"""
import os

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'


class OCRRecognition(object):
    def __init__(self):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model = 'iic/cv_convnextTiny_ocr-recognition-licenseplate_damo'
        # model = './output/ocr_workdir_v2/output'
        self.ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model=model
        )

    def predict(self, img):
        return self.ocr_recognition(img)

    def __call__(self, img, **kwargs):
        return self.predict(img)


def interface01():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines.cv.ocr_recognition_pipeline import OCRRecognitionPipeline
    from modelscope.models.cv.ocr_recognition.model import OCRRecognition
    from modelscope.models.cv.ocr_recognition.preprocessor import OCRRecognitionPreprocessor

    model = 'iic/cv_convnextTiny_ocr-recognition-licenseplate_damo'
    ocr_recognition = pipeline(
        Tasks.ocr_recognition,
        model=model
    )
    print(type(ocr_recognition))
    print(type(ocr_recognition.model))
    print(type(ocr_recognition.preprocessor))
    print(ocr_recognition.model)

    ### 使用url
    img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_licenseplate//ocr_recognition_licenseplate.jpg'
    img_url = "../datasets/plates/沪A8C871.jpg"
    result = ocr_recognition(img_url)
    print(result)


def interface02():
    ocr_rec = OCRRecognition()

    img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_licenseplate//ocr_recognition_licenseplate.jpg'
    r1 = ocr_rec(img_url)
    print(r1)

    img_url = "../datasets/plates/沪A8C871.jpg"
    r2 = ocr_rec(img_url)
    print(r2)

    print("=" * 100)
    _dir = "../datasets/plates"
    for _name in os.listdir(_dir):
        _file = os.path.join(_dir, _name)
        _result = ocr_rec(_file)
        print(_name, _result)

if __name__ == '__main__':
    # interface01()
    interface02()
