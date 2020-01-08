from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import os

image_path = 'data' + os.sep +'test2.jpg'
image = io.imread(image_path)
# io.imshow(image)
# io.show()           #读入图片，展示

rows = image.shape[0]
cols = image.shape[1]
print(rows,cols)
image = image.reshape(rows * cols, 3)    #将原有的二维（长宽，颜色0——255） 转换为一维（0-255）

kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)    #n_init是初始运算10次，选出好的初始点位置
kmeans.fit(image)

cluster = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)   #建立一个ndarry，值为128个质心的值

labels = np.array(kmeans.labels_,dtype=np.uint8)    #将一维化的图形矩阵贴上0-128的标签

labels.reshape(rows,cols)       #再将像素点排布恢复，但从rgb变成了灰度值
print(labels.shape)

np.save('codebook_test.npy', cluster)
io.imsave('compress.jpg', labels)