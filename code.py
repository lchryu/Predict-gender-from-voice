import pandas as pd # đọc file dữ liệu
import numpy as np # làm việc với ma trận và array
from sklearn.model_selection import train_test_split # chia tập dữ liệu thành 2 phần train data & test data
from sklearn import decomposition # giảm chiều dữ liệu
from sklearn.tree import DecisionTreeClassifier # cây phân lớp
from sklearn.metrics import confusion_matrix # Ma trận nhầm lẫn
from sklearn.metrics import accuracy_score # Độ chính xác
import tkinter as inp # giao diện
import matplotlib.pyplot as plt # vẽ đồ thị
from tkinter.ttk import *
import seaborn as sns # tạo biểu đồ trực quan hoá dữ liệu
import os # clear console mỗi lần chạy lại chương trình
from functional import Predict_from_user_data # import GUI from functional.py

sns.set()
os.system("cls") # clear toàn bộ màn hình console

df = pd.read_csv("voice.csv") # đọc dữ liệu

# print(df.info()) # Thông tin tập dữ liệu của bài toán

''' 
count: Đếm số quan sát ko NA/null
mean: trung bình của các giá trị 
std: độ lệch chuẩn của các quan sát
min: giá trị tối thiểu trong đối tượng 
'''
# print(df.describe()) # Thống kê trong dữ liệu


# # 1.Kiểm tra tập dữ liệu có bị thiếu giá trị không
# import missingno as msno
# p = msno.bar(df)
# plt.show()


# # 2.Kiểm tra tập dữ liệu nhãn (Nhãn 0 có 1584 dữ liệu, nhãn 1 có 1584 dữ liệu)
# sns.countplot(y=df.label, data=df)
# plt.xlabel("Count of each Target class")
# plt.ylabel("Target classes")
# plt.show()




X = np.array(df.drop(columns=['label'])) # Ma trận dữ liệu X
y = np.array([df["label"]]).T # Ma trận nhãn lớp y

# Chọn tập có n thuộc tính tốt nhất bằng phương pháp pca
n = 0
score = 0
for i in range(1,21):
	print("Lan", i)
	pca = decomposition.PCA(n_components=i)
	pca.fit(X)
	# print(pca)
	Xbar = pca.transform(X)  # Áp dụng giảm kích thước cho X
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

# Dùng tập n thuộc tính tốt đã chọn để tạo ra tập train & tập test mới
print("N_components:", n)
main_pca = decomposition.PCA(n_components=n)
main_pca.fit(X)
Xbar1 = main_pca.transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(Xbar1, y, test_size=0.3, shuffle=True)

# Use CART to train model
mainModel = DecisionTreeClassifier(criterion = "gini")
mainModel.fit(X_train1, y_train1)
y_pred1 = mainModel.predict(X_test1)


cnf_matrix = confusion_matrix(y_test1, y_pred1)
print('Confusion matrix:')
print(cnf_matrix)

# ma tran nham lan
#                                        predict
#                      |     positive        |    negative
#    ------------------|---------------------|--------------
#      true | positive |  True positive (TP) | False Negative (FN)
#           | negative |  False positive (FP)| True Negative (TN)

# confusion matrix to precision + recall
def cm2pr_binary(cm):
    p = cm[0,0]/np.sum(cm[:,0])
    r = cm[0,0]/np.sum(cm[0])
    return (p, r)

# model evaluation
acc = accuracy_score(y_test1, y_pred1) #do cxac
# precision = (TP)/(TP+FP) ti le so diem true positive (TP) trong nhung diem duoc phan loai positive (TP+FP)
# recall = (TP)/(TP+FN) ti le so diem true positive (TP) trong nhung diem thuc su la positive (TP+FN)
precision,recall = cm2pr_binary(cnf_matrix)
# f1-score is a combination of precision and recall
f1_score = (2 * precision * recall) / (precision + recall)

print('Accuracy = {0:.2f}'.format(acc))
print('Precision = {0:.2f}'.format(precision))
print('Recall = {0:.2f}'.format(recall))
print('F1-score = {0:.2f}'.format(f1_score))

#Input data from the user side and then predict male or female
Predict_from_user_data(main_pca, mainModel)
