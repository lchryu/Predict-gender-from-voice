
inp.Label(master, text="Thiếu máu (0-không, 1-có):").grid(row=1)
e2 = Entry(master, width=30)
e2.grid(row=1, column=1)

inp.Label(master, text="Enzym CPK trong máu (mcg/l):").grid(row=2)
e3 = Entry(master, width=30)
e3.grid(row=2, column=1)

inp.Label(master, text="Tiểu đường (0-không, 1-có):").grid(row=3)
e4 = Entry(master, width=30)
e4.grid(row=3, column=1)

inp.Label(master, text="Lượng máu rời khỏi tim mỗi lần co bóp (%):").grid(row=4)
e5 = Entry(master, width=30)
e5.grid(row=4, column=1)

inp.Label(master, text="Cao huyết áp (0-không, 1-có):").grid(row=5)
e6 = Entry(master, width=30)
e6.grid(row=5, column=1)

inp.Label(master, text="Tiểu cầu trong máu (kilophat tiểu cầu / ml):").grid(row=6)
e7 = Entry(master, width=30)
e7.grid(row=6, column=1)

inp.Label(master, text="Huyết thanh creatinine trong máu (mg/dl):").grid(row=7)
e8 = Entry(master, width=30)
e8.grid(row=7, column=1)

inp.Label(master, text="Huyết thanh sodium trong máu (mEq/l):").grid(row=8)
e9 = Entry(master, width=30)
e9.grid(row=8, column=1)

inp.Label(master, text="Giới tính (0-nam, 1-nữ):").grid(row=9)
e10 = Entry(master, width=30)
e10.grid(row=9, column=1)

inp.Label(master, text="Hút thuốc (0-không, 1-có):").grid(row=10)
e11 = Entry(master, width=30)
e11.grid(row=10, column=1)

inp.Label(master, text="Thời gian theo dõi (ngày):").grid(row=11)
e12 = Entry(master, width=30)
e12.grid(row=11, column=1)

inp.Label(master, text="Dự đoán tử vong (0-không, 1-có)").grid(row=13)
e13 = Entry(master, width=30)
e13.grid(row=13, column=1)

def predict():
	age = int(e1.get())
	anaemia = int(e2.get())
	creatinine_phosphokinase = int(e3.get())
	diabetes = int(e4.get())
	ejection_fraction = int(e5.get())
	high_blood_pressure = int(e6.get())
	platelets = int(e7.get())
	serum_creatinine = float(e8.get())
	serum_sodium = int(e9.get())
	sex = int(e10.get())
	smoking = int(e11.get())
	time = int(e12.get())
	data_new = np.array([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
                                   high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
	data_new_pca = main_pca.transform(data_new)
	kq = mainModel.predict(data_new_pca)[0]
	e13.insert(0, kq)

inp.Button(master,text ="Dự đoán", command = predict,activebackground='green',
          justify='center').grid(row=12, column=1)