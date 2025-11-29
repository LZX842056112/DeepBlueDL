from kalman import *
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 虚拟计数线的起点和终点
line = [(0, 100), (2560, 100)]
# 车辆总数
counter = 0
# 正向车道的车辆数据（从下往上）
counter_up = 0
# 逆向车道的车辆数据（从上往下）
counter_down = 0

# 创建SORT跟踪器对象
tracker = Sort()
# 存储上一帧的跟踪结果，用于判断运动方向
memory = {}


# 线与线的碰撞检测：叉乘的方法判断两条线是否相交
# 计算叉乘符号
# 如果返回值 > 0：三点呈逆时针方向,如果返回值 < 0：三点呈顺时针方向
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
# 点C和D在线段AB的两侧, 点A和B在线段CD的两侧
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# 利用yoloV3模型进行目标检测
# 加载模型相关信息
# 加载可以检测的目标的类型
labelPath = "./yolo-coco/coco.names"
LABELS = open(labelPath).read().strip().split("\n")
# 生成多种不同的颜色
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
# 加载预训练的模型：权重 配置信息,进行恢复
weightsPath = "./yolo-coco/yoloV3.weights"
configPath = "./yolo-coco/yoloV3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# 获取yolo中每一层的名称
ln = net.getLayerNames()
# 获取输出层的名称: [yolo-82,yolo-94,yolo-106]
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# 获取输出层索引
unconnected_layers = net.getUnconnectedOutLayers()
if unconnected_layers.ndim == 1:
    # 调整索引
    ln = [ln[i - 1] for i in unconnected_layers]
else:
    ln = [ln[i[0] - 1] for i in unconnected_layers]

# 读取图像
# frame = cv2.imread('./images/car2.jpg')
# (W,H) = (None,None)
# (H,W) = frame.shape[:2]
# 视频
vs = cv2.VideoCapture('./input/test_1.mp4')
(W, H) = (None, None)  # 初始化视频宽高
writer = None  # 视频写入器
try:
    # 获取视频总帧数
    prop = cv2.cv.CV_CAP_PROP_Frame_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("INFO:{} total Frame in video".format(total))
except:
    print("[INFO] could not determine in video")

# 遍历每一帧图像
while True:
    (grabed, frame) = vs.read()
    if not grabed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # 将图像转换为blob,进行前向传播
    # 1/255.0：像素值归一化到0-1
    # swapRB=True：BGR转RGB（OpenCV用BGR，YOLO用RGB）
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # 将blob送入网络
    net.setInput(blob)
    start = time.time()
    # 前向传播，进行预测，返回目标框边界和相应的概率
    layerOutputs = net.forward(ln)
    end = time.time()

    # 存放目标的检测框
    boxes = []
    # 置信度
    confidences = []
    # 目标类别
    classIDs = []

    # 遍历每个输出
    for output in layerOutputs:
        # 遍历检测结果
        for detection in output:
            # detction:1*85 [5:]表示类别，[0:4]bbox的位置信息 【5】置信度
            scores = detection[5:]  # 获取各类别置信度
            classID = np.argmax(scores)  # 找到最高置信度的类别
            confidence = scores[classID]  # 获取该置信度值

            if confidence > 0.3:
                # 将检测结果与原图片进行适配
                # 将相对坐标转换为绝对坐标
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 左上角坐标
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                # 更新目标框，置信度，类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 非极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 检测框:左上角和右下角
    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == "car":  # 只处理汽车类别
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                dets.append([x, y, x + w, y + h, confidences[i]])
    # 类型设置
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    # # 显示
    # plt.imshow(frame[:,:,::-1])
    # plt.show()

    # SORT目标跟踪
    if np.size(dets) == 0:
        continue  # 如果没有检测到汽车，跳过
    else:
        tracks = tracker.update(dets)  # 更新跟踪器，获取跟踪结果
    boxes = []  # 当前帧的跟踪框
    indexIDs = []  # 跟踪目标的ID
    previous = memory.copy()  # 保存上一帧结果
    memory = {}  # 清空，用于存储当前帧

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])  # 框坐标
        indexIDs.append(int(track[4]))  # 目标ID
        memory[indexIDs[-1]] = boxes[-1]  # 存储到内存

    # 碰撞检测
    if len(boxes) > 0:
        i = int(0)
        # 遍历跟踪框
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            # 为每个ID分配唯一颜色
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            # 根据在上一帧和当前帧的检测结果，利用虚拟线圈完成车辆计数
            if indexIDs[i] in previous:  # 如果该目标在上一帧也存在
                previous_box = previous[indexIDs[i]]  # 获取上一帧位置
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                # 计算中心点
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))  # 上一帧中心
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))  # 当前帧中心

                # 利用p0,p1与line进行碰撞检测
                # 判断中心点连线是否与计数线相交
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                    # 判断行进方向
                    # 判断方向：比较上一帧和当前帧的y坐标
                    if y2 > y:  # 如果上一帧y更大，说明向上移动
                        counter_down += 1
                    else:  # 否则向下移动
                        counter_up += 1
            i += 1

    # 将车辆计数的相关结果放在视频上
    cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
    # 总数-蓝色
    cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
    # 上行-绿色
    cv2.putText(frame, str(counter_up), (130, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
    # 下行-红色
    cv2.putText(frame, str(counter_down), (230, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    # 将检测结果保存在视频
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码
        writer = cv2.VideoWriter("./output/output.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)  # 写入帧到输出视频
    cv2.imshow("", frame)  # 显示当前帧
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

"释放资源"
writer.release()  # 释放视频写入器
vs.release()  # 释放视频读取器
cv2.destroyAllWindows()  # 关闭所有窗口
