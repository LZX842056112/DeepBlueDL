# USAGE
# python test_recognizer.py --model fer2013/checkpoints/epoch_75.hdf5

# import the necessary packages
from config import emotion_config as config
from pyimage.preprocessing import ImageToArrayPreprocessor
from pyimage.io import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

# 加载模型：命令行参数指定
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	help="path to model checkpoint to load")
args = vars(ap.parse_args())

# initialize the testing data generator and image preprocessor
# 图像的像素进行归一化
testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
# 获取测试数据
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	aug=testAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# 加载模型
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# 预测
(loss, acc) = model.evaluate_generator(
	testGen.generator(),
	steps=testGen.numImages // config.BATCH_SIZE,
	max_queue_size=10)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

# close the testing database
# 读取数据流关闭
testGen.close()