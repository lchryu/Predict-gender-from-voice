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

inp.Label(master, text="peakf").grid(row=12)
e13 = Entry(master, width=30)
e13.grid(row=12, column=1)

inp.Label(master, text="meanfun").grid(row=13)
e14 = Entry(master, width=30)
e14.grid(row=13, column=1)

inp.Label(master, text="meanfun").grid(row=14)
e15 = Entry(master, width=30)
e15.grid(row=14, column=1)

inp.Label(master, text="maxfun").grid(row=15)
e16 = Entry(master, width=30)
e16.grid(row=15, column=1)

inp.Label(master, text="meandom").grid(row=16)
e17 = Entry(master, width=30)
e17.grid(row=16, column=1)

inp.Label(master, text="mindom").grid(row=17)
e18 = Entry(master, width=30)
e18.grid(row=17, column=1)

inp.Label(master, text="maxdom").grid(row=18)
e20 = Entry(master, width=30)
e20.grid(row=18, column=1)

inp.Label(master, text="dfrange").grid(row=19)
e21 = Entry(master, width=30)
e21.grid(row=19, column=1)

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
master.mainloop()
