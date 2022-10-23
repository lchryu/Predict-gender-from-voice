import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
import os
os.system("cls")
sns.set()
import matplotlib.pyplot as plt

voice = pd.read_csv('voice.csv')
# print(voice.info())
# print(voice.describe())

voice.label.replace({'male': 0, 'female': 1}, inplace=True)
# print(voice.head())

## null count analysis before modelling to keep check
# Kiểm tra tập dữ liệu có bị thiếu giá trị không
# import missingno as msno
# p = msno.bar(voice)
# plt.show()


# Kiểm tra tập dữ liệu nhãn (Nhãn 0 có 1584 dữ liệu, nhãn 1 có 1584 dữ liệu)
sns.countplot(y=voice.label, data=voice)
plt.xlabel("Count of each Target class")
plt.ylabel("Target classes")
plt.show()


#Without PCA
X = voice.drop(['label'], axis=1)
Y = voice.label
# print(X)
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier(criterion='gini')
cart.fit(x_train, y_train)
kq = cart.predict(x_test)
# print(kq)
print("Accuracy_CART:", cart.score(x_test, y_test))
print('MAE_CART:', metrics.mean_absolute_error(y_test, kq))
print('MSE_CART:', metrics.mean_squared_error(y_test, kq))
print('RMSE_CART:', np.sqrt(metrics.mean_squared_error(y_test, kq)))
print("confusion_matrix: ")
print(metrics.confusion_matrix(y_test, kq))
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: ', precision_score(y_test, kq, average='macro'))
print('Recall:', recall_score(y_test, kq))
print('F1_score:', f1_score(y_test, kq))

print()

print("---------------------------------------------------------------------------------")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20, stratify=Y)

from sklearn.decomposition import PCA
pca = PCA()


X_new = pca.fit_transform(X)

# tính toán phương sai
pca.get_covariance()
# print(X_new)

# phan trăm phuong sai
explained_variance = pca.explained_variance_ratio_
# print(explained_variance)

pca = PCA(n_components=15)

x_new = pca.fit_transform(X)
# print(x_new.shape)
# print(x_new)

x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_new, Y, test_size=0.3, random_state=20)
cart_new = DecisionTreeClassifier(criterion='gini')
cart_new.fit(x_train_new, y_train_new)
kq2 = cart_new.predict(x_test_new)
# # print(kq)
print("Accuracy_CART_PCA:", cart_new.score(x_test_new, y_test_new))
print('MAE_CART_PCA:', metrics.mean_absolute_error(y_test_new, kq2))
print('MSE_CART_PCA:', metrics.mean_squared_error(y_test_new, kq2))
print('RMSE_CART_PCA:', np.sqrt(metrics.mean_squared_error(y_test_new, kq2)))
print("confusion_matrix: ")
print(metrics.confusion_matrix(y_test_new, kq2))
print('Precision_new:', precision_score(y_test_new, kq2))
print('Recall_new:', recall_score(y_test_new, kq2))
print('F1_score_new:', f1_score(y_test_new, kq2))



# do thi perceptron
# ax1 = sns.distplot(y_test, hist=False, label='Thuc te',  color='r')
# sns.distplot(kq, label='Du doan', hist=False, color='b', ax=ax1)
# plt.show()

