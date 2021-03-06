# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 22:32:52 2018

@author: User
"""
import matplotlib.pyplot as plt
from sklearn import datasets ,svm # svm for svc classifier

digits = datasets.load_digits()# 讀入資料:digits為一個dict型別資料

from sklearn.preprocessing import scale

data = scale(digits.data) # 將資料標準化使用scale()

from sklearn.cross_validation import train_test_split
# 分割train data ,test data
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

# 使用SVC分類器
model = svm.SVC(gamma = 0.001, C = 100, kernel = 'linear')
model.fit(X_train,y_train)

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
# PCA降維
# Create a regular PCA model 
pca = PCA(n_components=3)
# n_components: PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n

# Fit and transform the data to the model
# Re_Dimension = pca.fit_transform(X_train)
Re_Dimension = Isomap(n_neighbors=10).fit_transform(X_train)#主成份分析不同的地方是 Isomap 屬於非線性的降維方法。

predicted = model.predict(X_train)

print('準確率:',svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test))

# 在 1x2 的網格上繪製子圖形
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# 調整外觀
fig.subplots_adjust(top=0.85)

# 繪製散佈圖 
ax[0].scatter(Re_Dimension[:, 0], Re_Dimension[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(Re_Dimension[:, 0], Re_Dimension[:, 1], c=y_train)
ax[1].set_title('Actual Labels')

# 標題
fig.suptitle('Predicted VS actual labels', fontsize=14, fontweight='bold')

plt.show()