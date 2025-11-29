# import the necessary packages
import h5py
import os

class HDF5DatasetWriter:
	# 初始化：dims指定存储数据的shape【N,H,W】, hdf文件的位置
	def __init__(self, dims, outputPath, dataKey="images",
		bufSize=1000):
		# check to see if the output path exists, and if so, raise
		# an exception
		# 判断输出路径是否存在，若存在
		if os.path.exists(outputPath):
			raise ValueError("The supplied `outputPath` already "
				"exists and cannot be overwritten. Manually delete "
				"the file before continuing.", outputPath)

		# open the HDF5 database for writing and create two datasets:
		# one to store the images/features and another to store the
		# class labels
		# 打开HDF文件
		self.db = h5py.File(outputPath, "w")
		# 创建存储图像和label的空间
		self.data = self.db.create_dataset(dataKey, dims,
			dtype="float")
		self.labels = self.db.create_dataset("labels", (dims[0],),
			dtype="int")

		# store the buffer size, then initialize the buffer itself
		# along with the index into the datasets
		# 缓存大小
		self.bufSize = bufSize
		# 缓存的内容
		self.buffer = {"data": [], "labels": []}
		self.idx = 0

	# 获取数据，存在buffer
	def add(self, rows, labels):
		# add the rows and labels to the buffer
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)

		# check to see if the buffer needs to be flushed to disk
		# 缓存size>阈值
		if len(self.buffer["data"]) >= self.bufSize:
			self.flush()
	# 将数据写入磁盘
	def flush(self):
		# write the buffers to disk then reset the buffer
		# 获取idx
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		self.buffer = {"data": [], "labels": []}
	# 存储数据中的label:表情的class
	def storeClassLabels(self, classLabels):
		# create a dataset to store the actual class label names,
		# then store the class labels
		dt = h5py.special_dtype(vlen=str) # `vlen=unicode` for Py2.7
		labelSet = self.db.create_dataset("label_names",
			(len(classLabels),), dtype=dt)
		labelSet[:] = classLabels
	# 关闭写入和读取的流
	def close(self):
		# check to see if there are any other entries in the buffer
		# that need to be flushed to disk
		if len(self.buffer["data"]) > 0:
			self.flush()

		# close the dataset
		self.db.close()