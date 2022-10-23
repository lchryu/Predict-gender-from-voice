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
	print("Score:", trained_score)
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

# # ma tran nham lan
# #                                        predict
# #                      |     positive        |    negative
# #    ------------------|---------------------|--------------
# #      true | positive |  True positive (TP) | False Negative (FN)
# #           | negative |  False positive (FP)| True Negative (TN)

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

# giao dien
master = inp.Tk()
master.title('Nhập thông tin: ')

inp.Label(master, text="meanfreq:").grid(row=0)
e1 = Entry(master, width=30)
e1.grid(row=0, column=1)

inp.Label(master, text="sd").grid(row=1)
e2 = Entry(master, width=30)
e2.grid(row=1, column=1)

inp.Label(master, text="median").grid(row=2)
e3 = Entry(master, width=30)
e3.grid(row=2, column=1)

inp.Label(master, text="Q25").grid(row=3)
e4 = Entry(master, width=30)
e4.grid(row=3, column=1)

inp.Label(master, text="Q75").grid(row=4)
e5 = Entry(master, width=30)
e5.grid(row=4, column=1)

inp.Label(master, text="IQR").grid(row=5)
e6 = Entry(master, width=30)
e6.grid(row=5, column=1)

inp.Label(master, text="skew").grid(row=6)
e7 = Entry(master, width=30)
e7.grid(row=6, column=1)

inp.Label(master, text="kurt").grid(row=7)
e8 = Entry(master, width=30)
e8.grid(row=7, column=1)

inp.Label(master, text="sp.ent").grid(row=8)
e9 = Entry(master, width=30)
e9.grid(row=8, column=1)

inp.Label(master, text="sfm").grid(row=9)
e10 = Entry(master, width=30)
e10.grid(row=9, column=1)

inp.Label(master, text="mode").grid(row=10)
e11 = Entry(master, width=30)
e11.grid(row=10, column=1)

inp.Label(master, text="centroid").grid(row=11)
e12 = Entry(master, width=30)
e12.grid(row=11, column=1)

inp.Label(master, text="meanfun").grid(row=12)
e13 = Entry(master, width=30)
e13.grid(row=12, column=1)

inp.Label(master, text="minfun").grid(row=13)
e14 = Entry(master, width=30)
e14.grid(row=13, column=1)

inp.Label(master, text="maxfun").grid(row=14)
e15 = Entry(master, width=30)
e15.grid(row=14, column=1)

inp.Label(master, text="meandom").grid(row=15)
e16 = Entry(master, width=30)
e16.grid(row=15, column=1)

inp.Label(master, text="mindom").grid(row=16)
e17 = Entry(master, width=30)
e17.grid(row=16, column=1)

inp.Label(master, text="maxdom").grid(row=17)
e18 = Entry(master, width=30)
e18.grid(row=17, column=1)

inp.Label(master, text="dfrange").grid(row=18)
e19 = Entry(master, width=30)
e19.grid(row=18, column=1)
#
inp.Label(master, text="modindx").grid(row=19)
e20 = Entry(master, width=30)
e20.grid(row=19, column=1)

inp.Label(master, text="Dự đoán: ").grid(row=21)
e21 = Entry(master, width=30)
e21.grid(row=21, column=1)

def predict():
	x1 = float(e1.get())
	x2 = float(e2.get())
	x3 = float(e3.get())
	x4 = float(e4.get())
	x5 = float(e5.get())
	x6 = float(e6.get())
	x7 = float(e7.get())
	x8 = float(e8.get())
	x9 = float(e9.get())
	x10 = float(e10.get())
	x11 = float(e11.get())
	x12 = float(e12.get())
	x13 = float(e13.get())
	x14 = float(e14.get())
	x15 = float(e15.get())
	x16 = float(e16.get())
	x17 = float(e17.get())
	x18 = float(e18.get())
	x19 = float(e19.get())
	x20 = float(e20.get())
	data_new = np.array([[x1,x2,x3,x4,x5,
                                   x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20]])
	data_new_pca = main_pca.transform(data_new)
	kq = mainModel.predict(data_new_pca)[0]
	e21.insert(0, kq)

inp.Button(master,text ="Dự đoán", command = predict,activebackground='green',
          justify='center').grid(row=20, column=1)
master.mainloop()

