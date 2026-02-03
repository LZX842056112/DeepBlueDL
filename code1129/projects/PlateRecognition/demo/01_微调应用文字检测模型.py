# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/28 20:16
Create User : 19410
Desc : 基于车牌数据训练车牌区域检测模型
训练参考文档：
    https://www.modelscope.cn/docs/sdk/model-training
    https://www.modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/summary
"""
import os

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'


def interface_v0():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines.cv.ocr_detection_pipeline import OCRDetectionPipeline
    from modelscope.models.cv.ocr_detection.preprocessor import OCRDetectionPreprocessor
    from modelscope.models.cv.ocr_detection.model import OCRDetection

    model = 'iic/cv_resnet18_ocr-detection-db-line-level_damo'
    ocr_detection = pipeline(
        Tasks.ocr_detection,
        model=model
    )
    print(ocr_detection)
    print(type(ocr_detection))
    print(ocr_detection.preprocessor)
    print(type(ocr_detection.model))
    print(ocr_detection.model)
    url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
    url = r'../datasets/ocr_detection.jpg'
    url = r'../datasets/plate_images/0.jpg'
    result = ocr_detection(url)
    print(result)

def finetune_training():
    ### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中
    ### 建议使用GPU环境
    import os
    from modelscope.metainfo import Trainers
    from modelscope.trainers import build_trainer
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.hub.snapshot_download import snapshot_download

    model_id = 'iic/cv_resnet18_ocr-detection-db-line-level_damo'
    cache_path = snapshot_download(model_id)  # 模型下载保存目录
    config_file = os.path.join(cache_path, 'configuration.json')  # 模型参数配置文件，可以自定义
    pretrained_model = os.path.join(cache_path, 'db_resnet18_public_line_640x640.pt')  # 预训练模型
    saved_dir = './workdirs/cv_resnet18_ocr-detection-db-line-level_damo'  # 训练结果保存目录
    saved_finetune_model = os.path.join(saved_dir, 'final.pt')  # 训练保存的模型路径
    saved_infer_model = os.path.join(saved_dir, 'pytorch_model.pt')  # 训练模型转换成推理模型的路径

    kwargs = dict(
        cfg_file=config_file,
        # gpu_ids=[
        #     0,
        # ],
        batch_size=8,
        max_epochs=5,
        base_lr=0.007,
        load_pretrain=True,
        pretrain_model=pretrained_model,
        cache_path=cache_path,
        train_data_dir=[
            './datasets/dbnet_finetune/custom_data/'
        ],
        train_data_list=[
            './datasets/dbnet_finetune/custom_data/train_list.txt'
        ],
        val_data_dir=[
            './datasets/dbnet_finetune/custom_data/'
        ],
        val_data_list=[
            './datasets/dbnet_finetune/custom_data/test_list.txt'
        ])
    trainer = build_trainer(
        name=Trainers.ocr_detection_db, default_args=kwargs
    )
    trainer.cfg.train.work_dir = saved_dir
    trainer.cfg.miscs.output_dir = saved_dir
    trainer.cfg.train.dataloader.num_workers = 0
    trainer.cfg.test.dataloader.num_workers = 0
    print(type(trainer))
    trainer.train()

def interface_v1():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines.cv.ocr_detection_pipeline import OCRDetectionPipeline
    from modelscope.models.cv.ocr_detection.preprocessor import OCRDetectionPreprocessor
    from modelscope.models.cv.ocr_detection.model import OCRDetection

    model = 'iic/cv_resnet18_ocr-detection-db-line-level_damo'
    model = "./workdirs/cv_resnet18_ocr-detection-db-line-level_damo"
    ocr_detection = pipeline(
        Tasks.ocr_detection,
        model=model
    )
    print(ocr_detection)
    print(type(ocr_detection))
    print(ocr_detection.preprocessor)
    print(type(ocr_detection.model))
    print(ocr_detection.model)
    url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg'
    url = r'../datasets/ocr_detection.jpg'
    url = r'../datasets/plate_images/0.jpg'
    result = ocr_detection(url)
    print(result)

if __name__ == '__main__':
    # interface_v0()
    # finetune_training()
    interface_v1()
