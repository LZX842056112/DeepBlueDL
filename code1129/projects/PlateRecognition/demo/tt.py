# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/29 14:54
Create User : 19410
Desc : xxx
"""

import cv2 as cv
import numpy as np

# 加载图像
img = cv.imread("2.jpg")
rows, cols, _ = img.shape

# 定义四个点
pts1 = np.float32([[15, 7], [120, 26], [120, 81], [13, 56]])
pts2 = np.float32([[0, 0], [300, 0], [300, 100], [0, 100]])
# M是一个3*3的矩阵
M = cv.getPerspectiveTransform(pts1, pts2)
print(M)
# 计算规则：src(x,y)=dst(m11x+m12y+m13/(m31x+m32y+m33), m21x+m22y+m23/(m31x+m32y+m33))
dst = cv.warpPerspective(img, M, (300, 100))

cv.imwrite("2_2.jpg", dst)
cv.imshow('img', img)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()