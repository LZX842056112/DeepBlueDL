# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml \
#	--model output/epoch_75.hdf5

# 工具包
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# 命令行参数
ap = argparse.ArgumentParser()
# 指定人脸检测器
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
# 人脸表情分类器
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained emotion detector CNN")
# 获取的视频
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
# 加载人脸检测器
detector = cv2.CascadeClassifier(args["cascade"])
# 加载表情识别的检测器
model = load_model(args["model"])
# 表情
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised",
	"neutral"]

# if a video path was not supplied, grab the reference to the webcam
# 如果没有视频，开启摄像头
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
# 读取视频
else:
	camera = cv2.VideoCapture(args["video"])

# 遍历视频
while True:
	# grab the current frame
	# 获取帧图像
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	# 若果没有成功，就退出
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	# resize
	frame = imutils.resize(frame, width=300)
	# 灰度图
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# initialize the canvas for the visualization, then clone
	# the frame so we can draw on it
	# 全零数组，全黑图像
	canvas = np.zeros((220, 300, 3), dtype="uint8")
	# 复制图像
	frameClone = frame.copy()

	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	# 检测人脸，VJ框架
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# ensure at least one face was found before continuing
	# 判断是否检测到人脸
	if len(rects) > 0:
		# determine the largest face area
		# 筛选出面积最大的人脸
		rect = sorted(rects, reverse=True,
			key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
		# 坐标和宽高
		(fX, fY, fW, fH) = rect

		# extract the face ROI from the image, then pre-process
		# it for the network
		# 扣取人脸图像
		roi = gray[fY:fY + fH, fX:fX + fW]
		# resize
		roi = cv2.resize(roi, (48, 48))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		# [N,H,W,C]
		roi = np.expand_dims(roi, axis=0)

		# make a prediction on the ROI, then lookup the class
		# label
		# 预测
		preds = model.predict(roi)[0]
		# 表情标签
		label = EMOTIONS[preds.argmax()]

		# loop over the labels + probabilities and draw them
		# 遍历所有的表情
		for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
			# construct the label text
			# 构建txt
			text = "{}: {:.2f}%".format(emotion, prob * 100)

			# draw the label + probability bar on the canvas
			w = int(prob * 300)
			# 绘制红色的矩形
			cv2.rectangle(canvas, (5, (i * 35) + 5),
				(w, (i * 35) + 35), (0, 0, 255), -1)
			# 添加文字
			cv2.putText(canvas, text, (10, (i * 35) + 23),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45,
				(255, 255, 255), 2)

		# 添加表情
		cv2.putText(frameClone, label, (fX, fY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		# 画红框
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
			(0, 0, 255), 2)

	# show our classifications + probabilities
	# 显示
	cv2.imshow("Face", frameClone)
	cv2.imshow("Probabilities", canvas)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
# 释放资源
camera.release()
cv2.destroyAllWindows()