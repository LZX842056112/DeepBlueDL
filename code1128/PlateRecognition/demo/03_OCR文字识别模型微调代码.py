# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/28 21:33
Create User : 19410
Desc : xxx
"""
import os

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'


def finetune_train01():
    from modelscope.hub.snapshot_download import snapshot_download
    from modelscope.metainfo import Trainers
    from modelscope.msdatasets import MsDataset
    from modelscope.trainers import build_trainer
    from modelscope.utils.config import Config, ConfigDict
    from modelscope.utils.constant import ModelFile, DownloadMode

    ### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中

    model_id = 'iic/cv_convnextTiny_ocr-recognition-licenseplate_damo'
    cache_path = snapshot_download(model_id)  # 模型下载保存目录
    config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)  # 模型参数配置文件，支持自定义
    cfg = Config.from_file(config_path)

    # 构建数据集，支持自定义
    train_data_cfg = ConfigDict(
        name='ICDAR13_HCTR_Dataset',
        split='test',
        namespace='damo',
        test_mode=False
    )

    train_dataset = MsDataset.load(
        dataset_name=train_data_cfg.name,
        split=train_data_cfg.split,
        namespace=train_data_cfg.namespace,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

    test_data_cfg = ConfigDict(
        name='ICDAR13_HCTR_Dataset',
        split='test',
        namespace='damo',
        test_mode=True
    )

    test_dataset = MsDataset.load(
        dataset_name=test_data_cfg.name,
        split=test_data_cfg.split,
        namespace=train_data_cfg.namespace,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

    # tmp_dir = tempfile.TemporaryDirectory().name  # 模型文件和log保存位置，默认为"work_dir/"
    tmp_dir = "./output/ocr_workdir_v1"

    # 自定义参数，例如这里将max_epochs设置为15，所有参数请参考configuration.json
    def _cfg_modify_fn(cfg):
        cfg.train.max_epochs = 15
        cfg.train.dataloader.batch_size_per_gpu = 4
        cfg.evaluation.dataloader.batch_size_per_gpu = 4
        return cfg

    ####################################################################################

    '''
    使用本地文件
        lmdb: 
            构建包含下列信息的lmdb文件 (key: value)
            'num-samples': 总样本数,
            'image-000000001': 图像的二进制编码,
            'label-000000001': 标签序列的二进制编码,
            ...
            image和label后的index为9位并从1开始
    下面为示例 (local_lmdb为本地的lmdb文件)
    '''

    # train_dataset = MsDataset.load(
    #     dataset_name=train_data_cfg.name,
    #     split=train_data_cfg.split,
    #     namespace=train_data_cfg.namespace,
    #     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    #     local_lmdb='./local_lmdb')

    # test_dataset = MsDataset.load(
    #     dataset_name=test_data_cfg.name,
    #     split=test_data_cfg.split,
    #     namespace=train_data_cfg.namespace,
    #     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    #     local_lmdb='./local_lmdb')

    ####################################################################################

    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        work_dir=tmp_dir,
        cfg_modify_fn=_cfg_modify_fn
    )

    # 模型训练
    trainer = build_trainer(name=Trainers.ocr_recognition, default_args=kwargs)
    trainer.train()


def finetune_train02():
    from modelscope.hub.snapshot_download import snapshot_download
    from modelscope.metainfo import Trainers
    from modelscope.msdatasets import MsDataset
    from modelscope.trainers import build_trainer
    from modelscope.utils.config import Config, ConfigDict
    from modelscope.utils.constant import ModelFile, DownloadMode
    from modelscope.models.cv.ocr_recognition.preprocessor import OCRRecognitionPreprocessor
    from modelscope.msdatasets.dataset_cls.custom_datasets import OCRRecognitionDataset

    ### 请确认您当前的modelscope版本，训练/微调流程在modelscope==1.4.0及以上版本中

    model_id = 'iic/cv_convnextTiny_ocr-recognition-licenseplate_damo'
    cache_path = snapshot_download(model_id)  # 模型下载保存目录
    config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)  # 模型参数配置文件，支持自定义
    cfg = Config.from_file(config_path)
    preprocessor = OCRRecognitionPreprocessor(model_dir=cache_path)

    # 构建数据集，支持自定义
    train_data_cfg = ConfigDict(
        name='ICDAR13_HCTR_Dataset',
        split='test',
        namespace='damo',
        test_mode=False
    )

    train_dataset = MsDataset.load(
        dataset_name=train_data_cfg.name,
        split=train_data_cfg.split,
        namespace=train_data_cfg.namespace,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    train_dataset = OCRRecognitionDataset(
        preprocessor=preprocessor,
        **train_dataset.config_kwargs
    )

    test_data_cfg = ConfigDict(
        name='ICDAR13_HCTR_Dataset',
        split='test',
        namespace='damo',
        test_mode=True
    )

    test_dataset = MsDataset.load(
        dataset_name=test_data_cfg.name,
        split=test_data_cfg.split,
        namespace=train_data_cfg.namespace,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    test_dataset = OCRRecognitionDataset(
        preprocessor=preprocessor,
        **test_dataset.config_kwargs
    )

    # todo: 临时控制样本数量的逻辑
    train_dataset.nSamples = 50
    test_dataset.nSamples = 20

    # tmp_dir = tempfile.TemporaryDirectory().name  # 模型文件和log保存位置，默认为"work_dir/"
    tmp_dir = "./output/ocr_workdir_v2"

    # 自定义参数，例如这里将max_epochs设置为15，所有参数请参考configuration.json
    def _cfg_modify_fn(cfg):
        cfg.train.max_epochs = 15
        cfg.train.dataloader.batch_size_per_gpu = 4
        cfg.evaluation.dataloader.batch_size_per_gpu = 4
        return cfg

    ####################################################################################

    '''
    使用本地文件
        lmdb: 
            构建包含下列信息的lmdb文件 (key: value)
            'num-samples': 总样本数,
            'image-000000001': 图像的二进制编码,
            'label-000000001': 标签序列的二进制编码,
            ...
            image和label后的index为9位并从1开始
    下面为示例 (local_lmdb为本地的lmdb文件)
    '''

    # train_dataset = MsDataset.load(
    #     dataset_name=train_data_cfg.name,
    #     split=train_data_cfg.split,
    #     namespace=train_data_cfg.namespace,
    #     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    #     local_lmdb='./local_lmdb')

    # test_dataset = MsDataset.load(
    #     dataset_name=test_data_cfg.name,
    #     split=test_data_cfg.split,
    #     namespace=train_data_cfg.namespace,
    #     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    #     local_lmdb='./local_lmdb')

    ####################################################################################

    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        work_dir=tmp_dir,
        cfg_modify_fn=_cfg_modify_fn
    )

    # 模型训练
    trainer = build_trainer(name=Trainers.ocr_recognition, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    finetune_train02()
