import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import seaborn as sns
from sklearn.cluster import KMeans

file_path=r"D:\Bilal folder\AIML\unsupervised\kmean\image processing\istockphoto-814423752-612x612.jpg"

original_img=imgplt.imread(file_path)
print(original_img)
print(original_img.shape)
print(original_img.ndim)
print(len(original_img))

h,w,c=original_img.shape

img_2d=np.reshape(original_img,(h*w,c))
print(img_2d)
print(img_2d.shape)
print(img_2d.ndim)

model=KMeans(n_clusters=10)
label=model.fit_predict(img_2d)
print(label)

rgb_code=model.cluster_centers_.round(0).astype(int)
print(rgb_code)

quantizedimg=np.reshape(rgb_code[label],(h,w,c))
print(quantizedimg.shape)
print(quantizedimg.ndim)
print(len(quantizedimg))


fig,axes=plt.subplots(ncols=2,nrows=1,figsize=(10,6))

axes[0].imshow(original_img)
axes[0].set_title("original image")

axes[1].imshow(quantizedimg)
axes[1].set_title("Quantized image")


plt.show()