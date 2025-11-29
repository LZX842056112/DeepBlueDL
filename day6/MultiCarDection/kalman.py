from __future__ import print_function
# 对for循环有姮好的效果
from numba import jit
import numpy as np
# 用于线性分配，匈牙利匹配的实现
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
# 使用卡尔曼滤波器
from filterpy.kalman import KalmanFilter


@jit  # 告诉计算机"请加速这个函数"
def iou(bb_test, bb_gt):
    """
    在两个box间计算IOU
    :param bb_test: box1 = [x1y1x2y2]
    :param bb_gt: box2 = [x1y1x2y2]
    :return: 交并比IOU
    """
    # 计算相交区域的坐标
    xx1 = np.maximum(bb_test[0], bb_gt[0])  # 两个框相交区域的左边界
    yy1 = np.maximum(bb_test[1], bb_gt[1])  # 相交区域的上边界
    xx2 = np.minimum(bb_test[2], bb_gt[2])  # 相交区域的右边界
    yy2 = np.minimum(bb_test[3], bb_gt[3])  # 相交区域的下边界
    # 计算相交区域的宽高
    w = np.maximum(0., xx2 - xx1)  # 相交区域的宽度（如果为负说明没重叠）
    h = np.maximum(0., yy2 - yy1)
    wh = w * h  # 相交面积
    # 计算并集面积 = 两个框面积之和 - 相交面积
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])  # 第一个框面积
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])  # 第二个框面积
    union = area1 + area2 - wh  # 并集面积

    return wh / union  # IOU = 相交面积/并集面积


def convert_bbox_to_z(bbox):
    """
    将[x1,y1,x2,y2]形式的检测框转为滤波器的状态表示形式[x,y,s,r]。其中x，y是框的中心，s是w*h，r是宽高比
    :param bbox: [x1,y1,x2,y2] 分别是左上角坐标和右下角坐标
    :return: [ x, y, s, r ] 4行1列，其中x,y是box中心位置的坐标，s是面积，r是纵横比w/h
    """
    w = bbox[2] - bbox[0]  # 宽度
    h = bbox[3] - bbox[1]  # 高度
    x = bbox[0] + w / 2.  # 中心x
    y = bbox[1] + h / 2.  # 中心y
    s = w * h  # 面积
    r = w / float(h)  # 宽高比
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    将[cx，cy，s，r]的目标框表示转为[x_min，y_min，x_max，y_max]的形式
    :param x:[ x, y, s, r ],其中x,y是box中心位置的坐标，s是面积，r
    :param score: 置信度
    :return:[x1,y1,x2,y2],左上角坐标和右下角坐标
    """
    w = np.sqrt(x[2] * x[3])  # 宽度 = √(面积 × 宽高比)
    h = x[2] / w  # 高度 = 面积 ÷ 宽度
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


"""
# 表示观测目标框bbox所对应的单个跟踪对像的内部状态
"""


class KalmanBoxTracker(object):
    count = 0  # 全局计数器，给每个跟踪器唯一ID

    def __init__(self, bbox):
        """
        初始化边界框和跟踪器
        :param bbox:
        """
        # 定义等速模型
        # 内部使用KalmanFilter，7个状态变量和4个观测输入
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # F是状态变换模型 - 描述物体如何运动
        # 新位置 = 旧位置 + 速度
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x新 = x旧 + vx
            [0, 1, 0, 0, 0, 1, 0],  # y新 = y旧 + vy
            [0, 0, 1, 0, 0, 0, 1],  # s新 = s旧 + vs
            [0, 0, 0, 1, 0, 0, 0],  # r新 = r旧
            [0, 0, 0, 0, 1, 0, 0],  # vx新 = vx旧
            [0, 0, 0, 0, 0, 1, 0],  # vy新 = vy旧
            [0, 0, 0, 0, 0, 0, 1]  # vs新 = vs旧
        ])
        # 观测矩阵 H - 我们只能观测到前4个状态,不能直接测量速度
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # 观测到x
            [0, 1, 0, 0, 0, 0, 0],  # 观测到y
            [0, 0, 1, 0, 0, 0, 0],  # 观测到s
            [0, 0, 0, 1, 0, 0, 0]  # 观测到r
        ])
        # R是观测函数
        self.kf.R[2:, 2:] *= 10.  # 观测噪声：对面积和宽高比的不确定性更大
        # P是协方差矩阵
        self.kf.P[4:, 4:] *= 1000.  # 初始协方差：对速度非常不确定
        self.kf.P *= 10.  # 整体不确定性放大
        # Q是过程噪声矩阵
        self.kf.Q[-1, -1] *= 0.01  # 过程噪声：面积变化很小
        self.kf.Q[4:, 4:] *= 0.01  # 过程噪声：速度变化很小
        # 内部状态估计
        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 用检测框初始化状态
        self.time_since_update = 0  # 多久没更新了
        self.id = KalmanBoxTracker.count  # 唯一ID
        KalmanBoxTracker.count += 1  # 下一个ID
        self.history = []  # 历史轨迹
        self.hits = 0  # 总命中次数
        self.hit_streak = 0  # 连续匹配次数
        self.age = 0  # 存在了多少帧

    def update(self, bbox):
        """
        使用观察到的目标框更新状态向量。filterpy.kalman.KalmanFilter.update 会根据观测修改内部状态估计self.kf.x。
        重置self.time_since_update，清空self.history。
        :param bbox:目标框
        :return:
        """
        self.time_since_update = 0  # 重置"失联计时器"
        self.history = []  # 清空历史
        self.hits += 1  # 总命中+1
        self.hit_streak += 1  # 连续命中+1
        self.kf.update(convert_bbox_to_z(bbox))  # 用新检测更新状态

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计。
        将预测结果追加到self.history。由于 get_state 直接访问 self.kf.x，所以self.history没有用到
        :return:
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0  # 防止面积变成负数
        self.kf.predict()  # 卡尔曼预测
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0  # 如果上一帧没匹配，重置连续命中
        self.time_since_update += 1  # 失联时间+1
        self.history.append(convert_x_to_bbox(self.kf.x))  # 记录历史
        return self.history[-1]  # 返回最新预测

    def get_state(self):
        """
        返回当前边界框估计值
        :return:
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    将检测框bbox与卡尔曼滤波器的跟踪框进行关联匹配
    :param detections:检测框
    :param trackers:跟踪框，即跟踪目标
    :param iou_threshold:IOU阈值
    :return:跟踪成功目标的矩阵：matchs
            新增目标的矩阵：unmatched_detections
            跟踪失败即离开画面的目标矩阵：unmatched_trackers
    """
    # 跟踪目标数量为0，直接构造结果
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # iou 不支持数组计算。逐个计算两两间的交并比，调用 linear_assignment 进行匹配
    # 构建IOU矩阵：每一行是一个检测，每一列是一个跟踪器
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    # 遍历目标检测的bbox集合，每个检测框的标识为d
    for d, det in enumerate(detections):
        # 遍历跟踪框（卡尔曼滤波器预测）bbox集合，每个跟踪框标识为t
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)  # 计算IOU

    # 通过匈牙利算法将跟踪框和检测框以[[d,t]...]的二维矩阵的形式存储在match_indices中
    # 为什么是负号：linear_assignment的输入是成本矩阵，IOU越大对应的分配代价应越小
    # matched_indices = linear_assignment(-iou_matrix)
    result = linear_sum_assignment(-iou_matrix)  # 找最大IOU的匹配
    matched_indices = np.array(list(zip(*result)))

    # 记录未匹配的检测框及跟踪框
    # 未匹配的检测框放入unmatched_detections中，表示有新的目标进入画面，要新增跟踪器跟踪目标
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:  # 没找到对象的检测
            unmatched_detections.append(d)
    # 未匹配的跟踪框放入unmatched_trackers中，表示目标离开之前的画面，应删除对应的跟踪器
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:  # 没找到对象的跟踪器
            unmatched_trackers.append(t)
    # 将匹配成功的跟踪框放入matches中
    matches = []
    for m in matched_indices:
        # 过滤掉IOU低的匹配，将其放入到unmatched_detections和unmatched_trackers
        if iou_matrix[m[0], m[1]] < iou_threshold:  # IOU太小
            unmatched_detections.append(m[0])  # 当成未匹配检测
            unmatched_trackers.append(m[1])  # 当成未匹配跟踪器
        # 满足条件的以[[d,t]...]的形式放入matches中
        else:
            matches.append(m.reshape(1, 2))  # 有效匹配
    # 初始化matches,以np.array的形式返回
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    """
    Sort 是一个多目标跟踪器，管理多个 KalmanBoxTracker 对象
    """

    def __init__(self, max_age=1, min_hits=3):
        """
        初始化：设置SORT算法的关键参数
        :param max_age: 最大检测数：目标未被检测到的帧数，超过之后会被删除
        :param min_hits:
        """
        self.max_age = max_age  # 最大失联帧数
        self.min_hits = min_hits  # 最小确认帧数
        self.trackers = []  # 活跃跟踪器列表
        self.frame_count = 0  # 总帧数

    def update(self, dets):
        """
        该方法实现了SORT算法，输入是当前帧中所有物体的检测框的集合，包括目标的score,
        输出是当前帧目标的跟踪框集合，包括目标的跟踪的id
        要求是即使检测框为空，也必须对每一帧调用此方法，返回一个类似的输出数组，最后一列是目标对像的id
        注意：返回的目标对象数量可能与检测框的数量不同
        :param dets:以[[x1,y1,x2,y2,score]，[x1,y1,x2,y2,score]，...]形式输入的numpy.array
        :return:
        """
        self.frame_count += 1  # 记录处理到第几帧了
        # 在当前帧逐个预测轨迹位置，记录状态异常的跟踪器索引
        # 根据当前所有的卡尔曼跟踪器个数（即上一帧中跟踪的目标个数）创建二维数组：行号为卡尔曼滤波器的标识索引，列向量为跟踪框的位置和ID
        trks = np.zeros((len(self.trackers), 5))  # 存储跟踪器的预测
        to_del = []  # 存储要删除的目标框
        ret = []  # 存储要返回的追踪目标框
        # 循环遍历卡尔曼跟踪器列表
        for t, trk in enumerate(trks):
            # 使用卡尔曼跟踪器t产生对应目标的跟踪框
            pos = self.trackers[t].predict()[0]
            # 遍历完成后，trk中存储了上一帧中跟踪的目标的预测跟踪框
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 如果跟踪框中包含空值则将该跟踪框添加到要删除的列表中
            if np.any(np.isnan(pos)):  # 预测结果异常
                to_del.append(t)
        # numpy.ma.masked_invalid 屏蔽出现无效值的数组（NaN 或 inf）
        # numpy.ma.compress_rows 压缩包含掩码值的2-D 数组的整行，将包含掩码值的整行去除
        # trks中存储了上一帧中跟踪的目标并且在当前帧中的预测跟踪框
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 删除异常预测
        # 逆向删除异常的跟踪器，防止破坏索引
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 将目标检测框与卡尔曼滤波器预测的跟踪框关联获取跟踪成功的目标，新增的目标，离开画面的目标
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # 将跟踪成功的目标框更新到对应的卡尔曼滤波器
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:  # 这个跟踪器匹配成功了
                d = matched[np.where(matched[:, 1] == t)[0], 0]  # 找到对应的检测
                # 使用观测的边界框更新状态向量
                trk.update(dets[d, :][0])

        # 为新增的目标创建新的卡尔曼滤波器对象进行跟踪
        for i in unmatched_dets:  # 未匹配的检测
            trk = KalmanBoxTracker(dets[i, :])  # 创建新跟踪器
            self.trackers.append(trk)

        # 自后向前遍历，仅返回在当前帧出现且命中周期大于self.min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于self.max_age则删除跟踪器。
        # hit_streak忽略目标初始的若干帧
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边界框的估计值
            d = trk.get_state()[0]  # 当前状态
            # 跟踪成功目标的box与id放入ret列表中
            # 只有可靠的跟踪器才输出
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # 跟踪失败或离开画面的目标从卡尔曼跟踪器中删除
            if trk.time_since_update > self.max_age:  # 跟踪器失联太久
                self.trackers.pop(i)  # 删除跟踪器
        # 返回当前画面中所有目标的box与id,以二维矩阵形式返回
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
