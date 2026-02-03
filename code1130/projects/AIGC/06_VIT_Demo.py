# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/30 10:30
Create User : 19410
Desc :

https://modelscope.cn/models/iic/cv_vit-base_image-classification_Dailylife-labels/summary

"""

import os

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# # https://huggingface.co/
# # https://hf-mirror.com/
# import transformers  # pip install transformers


def interface01():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/bird.JPEG'
    image_classification = pipeline(
        Tasks.image_classification,
        model='iic/cv_vit-base_image-classification_Dailylife-labels'
    )
    result = image_classification(img_path)
    print(result)

    print(type(image_classification))
    print(type(image_classification.preprocessor))
    print(type(image_classification.model))
    print(image_classification.model)

def interface02():
    # from transformers.models.vit.image_processing_vit import ViTImageProcessor
    # from transformers.models.vit.modeling_vit import ViTForImageClassification
    from modelscope import ViTImageProcessor, ViTForImageClassification
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    url = r"./dog.png"
    url = r"./car.jpg"
    if url.startswith("http"):
        url = requests.get(url, stream=True).raw
    image = Image.open(url).convert("RGB")

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    print(type(processor))
    print(type(model))
    print(model)


    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

if __name__ == '__main__':
    interface02()