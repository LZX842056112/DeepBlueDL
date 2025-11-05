import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow.keras as keras
import cv2
import pickle
import copy
from detection.datasets import transforms, utils

class pascal_voc(keras.utils.Sequence):
    def __init__(self, phase):
        # pascal_voc 2007数据的存储路径
        self.data_path = os.path.join('../VOCdevkit', 'VOC2007')
        # batch_size
        self.batch_size = 1
        # 图片的最小尺寸
        self.target_size = 800
        # 图片的最大尺寸
        self.max_size = 1024
        # 输入网络中的图像尺寸
        self.scale = (1024, 1024)
        # 类别信息  ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus'....]
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

        # 构建目标类别的字典{'background': 0, 'aeroplane': 1, "bicycle": 2....}
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        # 像素RGB的均值
        self.pixel_means = np.array([[[122.7717, 115.9465, 102.9801]]])
        # 用来指明获取训练集或者是验证集数据
        self.phase = phase
        # 获取图像数量，并加载相应的标签
        self.load_labels()
        # 目标总数量
        self.num_gtlabels = len(self.gt_labels)
        self.img_transform = transforms.ImageTransform(self.scale, self.pixel_means, [1.,1.,1.], 'fixed')
        self.bbox_transform = transforms.BboxTransform()
        self.flip_ratio=0.5

    def __len__(self):
        # 返回迭代次数
        return np.round(self.num_image/self.batch_size)

    def __getitem__(self, idx):
        # 获取当前batch的起始索引值
        i = idx * self.batch_size
        batch_images = []
        batch_imgmeta= []
        batch_box = []
        bacth_labels = []
        for c in range(self.batch_size):
            # 获取相应的图像
            imname = self.gt_labels[i+c]['imname']
            # 读取图像
            image = self.image_read(imname)
            # 获取原始图像的尺寸
            ori_shape = image.shape
            # 进行尺度调整后的图像及调整的尺度
            image, image_scale,image_shape= self.prep_im_for_blob(image, self.pixel_means, self.target_size, self.max_size)
            # 获取尺度变换后图像的尺寸
            pad_shape = image.shape
            # 将gt_boxlabel与scale相乘获取图像调整后的标注框的大小：boxes.shape=[num_obj,4]
            bboxes = self.gt_labels[i+c]['boxes'] * image_scale
            # 获取对应的类别信息
            labels = self.gt_labels[i+c]['gt_classs']
            # print(labels)
            # 图像的基本信息
            img_meta_dict = dict({
                'ori_shape': ori_shape,
                'img_shape': image_shape,
                'pad_shape': pad_shape,
                'scale_factor': image_scale
            })
            # 将字典转换为列表的形式
            image_meta = self.compose_image_meta(img_meta_dict)
            # print(image_meta)
            batch_images.append(image)
            bacth_labels.append(labels)
            batch_imgmeta.append(image_meta)
            batch_box.append(bboxes)

        # 将图像转换成tensorflow输入的形式:【batch_size,H,W,C】
        batch_images = np.reshape(batch_images, (self.batch_size, image.shape[0], image.shape[1], 3))
        # print(batch_images.shape)
        batch_imgmeta = np.reshape(batch_imgmeta,(self.batch_size,11))
        # print(batch_imgmeta.shape)
        batch_box = np.reshape(batch_box,(self.batch_size,bboxes.shape[0],4 ))
        # print(batch_box.shape)
        bacth_labels = np.reshape(bacth_labels,((self.batch_size,labels.shape[0])))
        # print(bacth_labels.shape)
        # 返回结果：尺度变换后的图像，图像元信息，目标框位置，目标类别
        return batch_images,batch_imgmeta, batch_box, bacth_labels

    def image_read(self, imname):
        # opencv 中默认图片色彩格式为BGR
        image = cv2.imread(imname)
        # 将图片转成RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image

    def load_labels(self):
        # 根据标签信息加载相应的数据
        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'val.txt')
        # 获取图像的索引
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
        self.num_image = len(self.image_index)
        # 图像对应的索引放到列表gt_labels中
        gt_labels = []
        # 遍历每一份图像获取标注信息
        for index in self.image_index:
            # 获取标注信息，包括objet box坐标信息 以及类别信息
            gt_label = self.load_pascal_annotation(index)
            # 添加到列表中
            gt_labels.append(gt_label)
        # 将标注信息赋值给属性：self.gt_labels
        self.gt_labels = gt_labels

    def load_pascal_annotation(self, index):
        """
        在PASCAL VOC的XML文件获取边框信息和类别信息
        """
        # 获取XML文件的地址
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        # 将XML中的内容获取出来
        tree = ET.parse(filename)
        # 获取节点图像的size
        image_size = tree.find('size')
        # 将图像的size信息存放到sizeinfo中
        size_info = np.zeros((2,), dtype=np.float32)
        # 宽
        size_info[0] = float(image_size.find('width').text)
        # 高
        size_info[1] = float(image_size.find('height').text)
        # 找到所有的object节点
        objs = tree.findall('object')
        # object的数量
        num_objs = len(objs)
        # boxes 坐标 (num_objs,4)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        # class 的数量num_objs个，每个目标一个类别
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # 遍历所有的目标
        for ix, obj in enumerate(objs):
            # 找到bndbox节点
            bbox = obj.find('bndbox')
            # 获取坐标框的位置信息
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            # 将位置信息存储在bbox中，注意boxes是一个np类的矩阵 大小为[num_objs,4]
            boxes[ix, :] = [y1, x1, y2, x2]
            # 找到class对应的类别信息
            cls = self.class_to_ind[obj.find('name').text.lower().strip()]
            # 将class信息存入gt_classses中，注意gt_classes也是一个np类的矩阵 大小为[num_objs] 是int值 对应于name
            gt_classes[ix] = cls
            # 获取图像的存储路径
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        # 返回结果
        return {'boxes': boxes, 'gt_classs': gt_classes, 'imname': imname,  'image_size': size_info,
                'image_index': index}

    def prep_im_for_blob(self, im, pixel_means, target_size, max_size):
        "对输入的图像进行处理"
        im = im.astype(np.float32, copy=False)
        # 减去均值
        im -= pixel_means
        # 图像的大小
        im_shape = im.shape
        # 最短边
        im_size_min = np.min(im_shape[0:2])
        # 最长边
        im_size_max = np.max(im_shape[0:2])
        # 短边变换到800的比例
        im_scale = float(target_size) / float(im_size_min)  # 600/最短边
        # 若长边以上述比例变换后大于1024，则修正变换比例
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        # 根据变换比例对图像进行resize
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        shape = (1024, 1024, im.shape[-1])
        pad = np.zeros(shape, dtype=im.dtype)
        pad[:im.shape[0], :im.shape[1], ...] = im
        # 返回im 和 im_scale
        return pad, im_scale, im.shape

    def compose_image_meta(self,img_meta_dict):
        '''将图像的原信息转换为列表.
        '''
        # 获取到字典中的信息
        ori_shape = img_meta_dict['ori_shape']
        img_shape = img_meta_dict['img_shape']
        pad_shape = img_meta_dict['pad_shape']
        scale_factor = img_meta_dict['scale_factor']
        flip = 0
        # 转换为一维列表
        img_meta = np.array(
            ori_shape +  # size=3
            img_shape +  # size=3
            pad_shape +  # size=3
            tuple([scale_factor]) + # size=1
            tuple([flip])  # size=1
        ).astype(np.float32)
        # 返回列表信息
        return img_meta


if __name__ == '__main__':
    import visualize
    pascal = pascal_voc('train')
    # print(pascal.gt_labels)
    image, image_meta, bboxes, labels = pascal[10]
    print(image.shape)
    print(image_meta.shape)
    # print (len(pascal.gt_labels))
    from detection.datasets.utils import get_original_image
    import matplotlib.pyplot as plt
    import cv2

    # img, img_meta, bboxes, labels = data[0].image
    img_mean = [[122.7717, 115.9465, 102.9801]]
    img_std = (1., 1., 1.)
    rgb_img = np.round(image + img_mean).astype('uint8')
    # ori_img = get_original_image(data['image'], img_meta, img_mean)

    visualize.display_instances(rgb_img[0], bboxes[0], labels[0], pascal.classes)
    plt.show()

