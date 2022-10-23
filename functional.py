import numpy as np
import tkinter as tkr
from tkinter import messagebox
from tkinter.ttk import *
import os
import random
def Predict_from_user_data(main_pca, mainModel):
    option = input("Bạn có muốn dự đoán giới tính từ dữ liệu người dùng nhập vào?(y/n): ")
    if (option == 'y'):
        # GUI
        win = tkr.Tk()
        win.attributes("-topmost", True) #luôn luôn ghim của sổ trên màn hình
        win.geometry("280x500")
        win.title('Nhập thông tin: ')

        tkr.Label(win, text="meanfreq:").grid(row=0)
        e1 = Entry(win, width=30)
        e1.grid(row=0, column=1)

        tkr.Label(win, text="sd").grid(row=1)
        e2 = Entry(win, width=30)
        e2.grid(row=1, column=1)

        tkr.Label(win, text="median").grid(row=2)
        e3 = Entry(win, width=30)
        e3.grid(row=2, column=1)

        tkr.Label(win, text="Q25").grid(row=3)
        e4 = Entry(win, width=30)
        e4.grid(row=3, column=1)

        tkr.Label(win, text="Q75").grid(row=4)
        e5 = Entry(win, width=30)
        e5.grid(row=4, column=1)

        tkr.Label(win, text="IQR").grid(row=5)
        e6 = Entry(win, width=30)
        e6.grid(row=5, column=1)

        tkr.Label(win, text="skew").grid(row=6)
        e7 = Entry(win, width=30)
        e7.grid(row=6, column=1)

        tkr.Label(win, text="kurt").grid(row=7)
        e8 = Entry(win, width=30)
        e8.grid(row=7, column=1)

        tkr.Label(win, text="sp.ent").grid(row=8)
        e9 = Entry(win, width=30)
        e9.grid(row=8, column=1)

        tkr.Label(win, text="sfm").grid(row=9)
        e10 = Entry(win, width=30)
        e10.grid(row=9, column=1)

        tkr.Label(win, text="mode").grid(row=10)
        e11 = Entry(win, width=30)
        e11.grid(row=10, column=1)

        tkr.Label(win, text="centroid").grid(row=11)
        e12 = Entry(win, width=30)
        e12.grid(row=11, column=1)

        tkr.Label(win, text="meanfun").grid(row=12)
        e13 = Entry(win, width=30)
        e13.grid(row=12, column=1)

        tkr.Label(win, text="minfun").grid(row=13)
        e14 = Entry(win, width=30)
        e14.grid(row=13, column=1)

        tkr.Label(win, text="maxfun").grid(row=14)
        e15 = Entry(win, width=30)
        e15.grid(row=14, column=1)

        tkr.Label(win, text="meandom").grid(row=15)
        e16 = Entry(win, width=30)
        e16.grid(row=15, column=1)

        tkr.Label(win, text="mindom").grid(row=16)
        e17 = Entry(win, width=30)
        e17.grid(row=16, column=1)

        tkr.Label(win, text="maxdom").grid(row=17)
        e18 = Entry(win, width=30)
        e18.grid(row=17, column=1)

        tkr.Label(win, text="dfrange").grid(row=18)
        e19 = Entry(win, width=30)
        e19.grid(row=18, column=1)
        #
        tkr.Label(win, text="modindx").grid(row=19)
        e20 = Entry(win, width=30)
        e20.grid(row=19, column=1)

        # tkr.Label(win, text="Dự đoán: ").grid(row=21)
        # e21 = Entry(win, width=30)
        # e21.grid(row=21, column=1)

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
            # e21.insert(0, kq)
            messagebox.showinfo("Predict", f"This person's gender is {kq}")
        def random1():
            e1.delete(0, tkr.END)
            e1.insert(0, random.random())
            e2.delete(0, tkr.END)
            e2.insert(0, random.random())
            e3.delete(0, tkr.END)
            e3.insert(0, random.random())
            e4.delete(0, tkr.END)
            e4.insert(0, random.random())
            e5.delete(0, tkr.END)
            e5.insert(0, random.random())
            e6.delete(0, tkr.END)
            e6.insert(0, random.random())
            e7.delete(0, tkr.END)
            e7.insert(0, random.random())
            e8.delete(0, tkr.END)
            e8.insert(0, random.random())
            e9.delete(0, tkr.END)
            e9.insert(0, random.random())
            e10.delete(0, tkr.END)
            e10.insert(0, random.random())
            e11.delete(0, tkr.END)
            e11.insert(0, random.random())
            e12.delete(0, tkr.END)
            e12.insert(0, random.random())
            e13.delete(0, tkr.END)
            e13.insert(0, random.random())
            e14.delete(0, tkr.END)
            e14.insert(0, random.random())
            e15.delete(0, tkr.END)
            e15.insert(0, random.random())
            e16.delete(0, tkr.END)
            e16.insert(0, random.random())
            e17.delete(0, tkr.END)
            e17.insert(0, random.random())
            e18.delete(0, tkr.END)
            e18.insert(0, random.random())
            e19.delete(0, tkr.END)
            e19.insert(0, random.random())
            e20.delete(0, tkr.END)
            e20.insert(0, random.random())
            
        tkr.Button(win,text ="Dự đoán", command = predict,activebackground='green',
                justify='center', bg = 'green', fg = 'white').grid(row=20, column=1)
        tkr.Button(win,text ="Random", command = random1,activebackground='green',
                justify='center', bg = 'green', fg = 'white').grid(row=22, column=1)
        win.mainloop()
    else:
        os.system("cls")
        print("Cảm ơn các bạn đã xem chương trình")