import pandas as pd #doc file
import numpy as np # lam viec voi ma tran & array
from sklearn.model_selection import train_test_split # chia tap dl thanh 2 phan traning data & test data
from sklearn import decomposition # giam chieu dl
from sklearn.tree import DecisionTreeClassifier # cay phan lop
from sklearn.metrics import confusion_matrix # ma tran nham lan
from sklearn.metrics import accuracy_score # do chinh xac
import tkinter as inp #giao dien
from tkinter.ttk import *
import os
from functional import Predict_from_user_data
os.system("cls")
# doc file
df = pd.read_csv("voice.csv")
# ma tran du lieu X
X = np.array(df.drop(columns=['label']))


# ma tran nhan lop Y
y = np.array([df["label"]]).T
# chon tap co n thuoc tinh tot nhat = phg phap PCA
n = 0
score = 0
for i in range(1,21):
	print("Lan", i)
	pca = decomposition.PCA(n_components=i)
	pca.fit(X)
	# print(pca)
	Xbar = pca.transform(X)  # ap dung giam kich thuoc cho X.
	X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=False)
	model = DecisionTreeClassifier(criterion = "gini")
	model = model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	d = 0
	trained_score = accuracy_score(y_test, y_pred)
	print("Score =", trained_score)
	if(trained_score >= score):
		score = trained_score
		n = i

# dung tap n thuoc tinh tot da chon de tao ra tap tran & test moi
print("N_components:", n)
main_pca = decomposition.PCA(n_components=n)
main_pca.fit(X)
Xbar1 = main_pca.transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(Xbar1, y, test_size=0.3, shuffle=True)

# dung CART (cay phan lop) de xd mo hinh
mainModel = DecisionTreeClassifier(criterion = "gini")
mainModel.fit(X_train1, y_train1)
y_pred1 = mainModel.predict(X_test1)


cnf_matrix = confusion_matrix(y_test1, y_pred1)
print('Confusion matrix:')
print(cnf_matrix)

# ham tinh precision va recall
def cm2pr_binary(cm):
    p = cm[0,0]/np.sum(cm[:,0])
    r = cm[0,0]/np.sum(cm[0])
    return (p, r)

# danh gia mo hinh
acc = accuracy_score(y_test1, y_pred1) #do cxac
# precision = ti le so diem true positive (TP) trong nhung diem duoc phan loai positive (TP+FP)
# recall = ti le so diem true positive (TP) trong nhung diem thuc su la positive (TP+FN)
precision,recall = cm2pr_binary(cnf_matrix)
# f1-score la ket hop cua precision & recall
f1_score = (2 * precision * recall) / (precision + recall)

print('Accuracy = {0:.2f}'.format(acc))
print('Precision = {0:.2f}'.format(precision))
print('Recall = {0:.2f}'.format(recall))
print('F1-score = {0:.2f}'.format(f1_score))

#Nhập dữ liệu từ phía người dùng rồi dự đoán là nam hay nữ 
Predict_from_user_data()
