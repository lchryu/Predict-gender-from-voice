import tkinter as inp

# GUI
master = inp.Tk()
master.title('Nhập thông tin: ')

inp.Label(master, text="meanfreq:").grid(row=0)
e1 = tkinter.Entry(master, width=30)
e1.grid(row=0, column=1)

inp.Label(master, text="sd").grid(row=1)
e2 = tkinter.Entry(master, width=30)
e2.grid(row=1, column=1)

inp.Label(master, text="median").grid(row=2)
e3 = tkinter.Entry(master, width=30)
e3.grid(row=2, column=1)

inp.Label(master, text="Q25").grid(row=3)
e4 = tkinter.Entry(master, width=30)
e4.grid(row=3, column=1)

inp.Label(master, text="Q75").grid(row=4)
e5 = tkinter.Entry(master, width=30)
e5.grid(row=4, column=1)

inp.Label(master, text="IQR").grid(row=5)
e6 = tkinter.Entry(master, width=30)
e6.grid(row=5, column=1)

inp.Label(master, text="skew").grid(row=6)
e7 = tkinter.Entry(master, width=30)
e7.grid(row=6, column=1)

inp.Label(master, text="kurt").grid(row=7)
e8 = tkinter.Entry(master, width=30)
e8.grid(row=7, column=1)

inp.Label(master, text="sp.ent").grid(row=8)
e9 = tkinter.Entry(master, width=30)
e9.grid(row=8, column=1)

inp.Label(master, text="sfm").grid(row=9)
e10 = tkinter.Entry(master, width=30)
e10.grid(row=9, column=1)

inp.Label(master, text="mode").grid(row=10)
e11 = tkinter.Entry(master, width=30)
e11.grid(row=10, column=1)

inp.Label(master, text="centroid").grid(row=11)
e12 = tkinter.Entry(master, width=30)
e12.grid(row=11, column=1)

inp.Label(master, text="meanfun").grid(row=12)
e13 = tkinter.Entry(master, width=30)
e13.grid(row=12, column=1)

inp.Label(master, text="minfun").grid(row=13)
e14 = tkinter.Entry(master, width=30)
e14.grid(row=13, column=1)

inp.Label(master, text="maxfun").grid(row=14)
e15 = tkinter.Entry(master, width=30)
e15.grid(row=14, column=1)

inp.Label(master, text="meandom").grid(row=15)
e16 = tkinter.Entry(master, width=30)
e16.grid(row=15, column=1)

inp.Label(master, text="mindom").grid(row=16)
e17 = tkinter.Entry(master, width=30)
e17.grid(row=16, column=1)

inp.Label(master, text="maxdom").grid(row=17)
e18 = tkinter.Entry(master, width=30)
e18.grid(row=17, column=1)

inp.Label(master, text="dfrange").grid(row=18)
e19 = tkinter.Entry(master, width=30)
e19.grid(row=18, column=1)
#
inp.Label(master, text="modindx").grid(row=19)
e20 = tkinter.Entry(master, width=30)
e20.grid(row=19, column=1)

inp.Label(master, text="Dự đoán: ").grid(row=21)
e21 = tkinter.Entry(master, width=30)
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
