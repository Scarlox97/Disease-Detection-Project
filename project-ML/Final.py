#Python Tkinter and Sqlite3 Login Form

#imports
from tkinter import *
from tkinter import messagebox as ms
import sqlite3
import pandas as pd
import csv
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import time

from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict
from sklearn.metrics import accuracy_score



# make database and users (if not exists already) table at programme start up
with sqlite3.connect('quit.db') as db:
    c = db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS user (username TEXT NOT NULL ,password TEX NOT NULL);')
db.commit()
db.close()

#main Class
class main:
    def __init__(self,master):
    	# Window 
        self.master = master
        # Some Usefull variables
        self.username = StringVar()
        self.password = StringVar()
        self.n_username = StringVar()
        self.n_password = StringVar()
        #Create Widgets
        self.widgets()

    #Login Function
    def login(self):
    	#Establish Connection
        with sqlite3.connect('quit.db') as db:
            c = db.cursor()

        #Find user If there is any take proper action
        find_user = ('SELECT * FROM user WHERE username = ? and password = ?')
        c.execute(find_user,[(self.username.get()),(self.password.get())])
        result = c.fetchall()
        if result:
            self.logf.pack_forget()
            self.head['text'] = self.username.get() + '\n Loged In'
            self.head['pady'] = 150
            self.visible()
        else:
            ms.showerror('Oops!','Username Not Found.')


    def visible(self):
        win=Tk()
        win.title("REDIRECTING")
        Button(win,text="Redirecting for Disease Detection",command=self.answer,width=30,bd = 3 ,font = ('',15),padx=5).grid(row=4,column=4)
        Button(win,text="Want to go back",command=self.cr,bd = 3 ,font = ('',15),padx=5).grid(row=2,column=4)

    def answer(self):
        print("hello world")
        #reading training csv file
        data = pd.read_csv("/home/mudit/project/Manual-Data/Training.csv")
        df = pd.DataFrame(data)
        cols = df.columns
        cols = cols[:-1]
 

        x = df[cols]
        y = df['prognosis']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        #reading testing csv file
        test_data = pd.read_csv("/home/mudit/project/Manual-Data/Testing.csv")
        testx = test_data[cols]
        testy = test_data['prognosis']
        #creating tkinter window
        master = Tk()
        master.title("Welcome to the Disease Detection Portal  ")
        blank = Entry(master,width=25,bd=2)
        blank.grid(row=20, column=18)
        def var_states():
                Label(master, text="Symptoms noticed :",font="arial 16 bold").grid(row=2, sticky=W)
                #splitting test data and train data
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
                print ("DecisionTree")
                #appliying decision tree classifier
                dt = DecisionTreeClassifier()
                #now fiting data into classifier
                clf_dt=dt.fit(x_train,y_train)
                global r
                r=np.array([var1.get(),var2.get(),var3.get(),var4.get(),var5.get(),var6.get(),var7.get(),var8.get(),var9.get(),var10.get(),var11.get(),var12.get(),var13.get(),var14.get(),var15.get(),var16.get(),var17.get(),var18.get(),var19.get(),var20.get(),var21.get(),var22.get(),var23.get(),var24.get(),var25.get(),var26.get(),var27.get(),var28.get(),var29.get(),var30.get(),var31.get(),var32.get(),var33.get(),var34.get(),var35.get(),var36.get(),var37.get(),var38.get(),var39.get(),var40.get(),var41.get(),var42.get(),var43.get(),var44.get(),var45.get(),var46.get(),var47.get(),var48.get(),var49.get(),var50.get(),var51.get(),var52.get(),var53.get(),var54.get(),var55.get(),var56.get(),var57.get(),var58.get(),var59.get(),var60.get(),var61.get(),var62.get(),var63.get(),var64.get(),var65.get(),var66.get(),var67.get(),var68.get(),var69.get(),var70.get(),var71.get(),var72.get(),var73.get(),var74.get(),var75.get(),var76.get(),var77.get(),var78.get(),var79.get(),var80.get(),var81.get(),var82.get(),var83.get(),var84.get(),var85.get(),var86.get(),var87.get(),var88.get(),var89.get(),var90.get(),var91.get(),var92.get(),var93.get(),var94.get(),var95.get(),var96.get(),var97.get(),var98.get(),var99.get(),var100.get(),var101.get(),var102.get(),var103.get(),var104.get(),var105.get(),var106.get(),var107.get(),var108.get(),var109.get(),var110.get(),var111.get(),var112.get(),var113.get(),var114.get(),var115.get(),var116.get(),var117.get(),var118.get(),var119.get(),var120.get(),var121.get(),var122.get(),var123.get(),var124.get(),var125.get(),var126.get(),var127.get(),var128.get(),var129.get(),var130.get(),var131.get(),var132.get()])
                #predicting output
                output=clf_dt.predict([r])
                #displaying output in blank space on tkinter window
                blank.delete(0,END)
                blank.insert(0,output)
        
  
        var1 = IntVar()
        Checkbutton(master, text="itching", variable=var1).grid(row=4, column=0, sticky=W)

        var2 = IntVar()
        Checkbutton(master, text="skin_rash", variable=var2).grid(row=4,column=1, sticky=W)

        var3 = IntVar()
        Checkbutton(master, text="nodal_skin_eduptions", variable=var3).grid(row=4,column=2, sticky=W)

        var4 = IntVar()
        Checkbutton(master, text="continous_sneezing", variable=var4).grid(row=4,column=3 ,sticky=W)

        var5 = IntVar()
        Checkbutton(master, text="shivering", variable=var5).grid(row=5,column=0, sticky=W)

        var6 = IntVar()
        Checkbutton(master, text="chills", variable=var6).grid(row=5,column=1, sticky=W)

        var7 = IntVar()
        Checkbutton(master, text="joint_pain", variable=var7).grid(row=5,column=2, sticky=W)

        var8 = IntVar()
        Checkbutton(master, text="stomach_pain", variable=var8).grid(row=5,column=3 ,sticky=W)

        var9 = IntVar()
        Checkbutton(master, text="acidity", variable=var9).grid(row=6,column=0, sticky=W)

        var10 = IntVar()
        Checkbutton(master, text="ulcer_on_tongue", variable=var10).grid(row=6,column=1, sticky=W)

        var11 = IntVar()
        Checkbutton(master, text="muscle_wasting", variable=var11).grid(row=6,column=2, sticky=W)

        var12 = IntVar()
        Checkbutton(master, text="vomiting", variable=var12).grid(row=6,column=3, sticky=W)

        var13 = IntVar()
        Checkbutton(master, text="burning_micturition", variable=var13).grid(row=7,column=0, sticky=W)

        var14 = IntVar()
        Checkbutton(master, text="spotting_urination", variable=var14).grid(row=7,column=1,sticky=W)

        var15 = IntVar()
        Checkbutton(master, text="fatigue", variable=var15).grid(row=7,column=2, sticky=W)

        var16 = IntVar()
        Checkbutton(master, text="weight_gain", variable=var16).grid(row=7,column=3, sticky=W)

        var17 = IntVar()
        Checkbutton(master, text="anxiety", variable=var17).grid(row=8,column=0 ,sticky=W)

        var18 = IntVar()
        Checkbutton(master, text="cold_hands_and_feets", variable=var18).grid(row=8,column=1 ,sticky=W)

        var19 = IntVar()
        Checkbutton(master, text="mood_swings", variable=var19).grid(row=8,column=2, sticky=W)

        var20 = IntVar()
        Checkbutton(master, text="weight_loss", variable=var20).grid(row=8,column=3 ,sticky=W)

        var21 = IntVar()
        Checkbutton(master, text="restlessness", variable=var21).grid(row=9, column=0,sticky=W)

        var22 = IntVar()
        Checkbutton(master, text="lethargy", variable=var22).grid(row=9,column=1, sticky=W)

        var23 = IntVar()
        Checkbutton(master, text="patches_in_throat", variable=var23).grid(row=9, column=2,sticky=W)

        var24 = IntVar()
        Checkbutton(master, text="irregular_sugar_level", variable=var24).grid(row=9,column=3, sticky=W)

        var25 = IntVar()
        Checkbutton(master, text="cough", variable=var25).grid(row=10,column=0, sticky=W)

        var26= IntVar()
        Checkbutton(master, text="high_fever", variable=var26).grid(row=10,column=1, sticky=W)

        var27 = IntVar()
        Checkbutton(master, text="sunken_eyes", variable=var27).grid(row=10,column=2, sticky=W)

        var28 = IntVar()
        Checkbutton(master, text="breathlessness", variable=var28).grid(row=10,column=3, sticky=W)

        var29 = IntVar()
        Checkbutton(master, text="sweating", variable=var29).grid(row=11,column=0, sticky=W)

        var30 = IntVar()
        Checkbutton(master, text="dehydration", variable=var30).grid(row=11, column=1,sticky=W)

        var31 = IntVar()
        Checkbutton(master, text="indigestion", variable=var31).grid(row=11,column=2, sticky=W)

        var32 = IntVar()
        Checkbutton(master, text="headache", variable=var32).grid(row=11,column=3, sticky=W)

        var33 = IntVar()
        Checkbutton(master, text="yellowish_skin", variable=var33).grid(row=12,column=0, sticky=W)

        var34 = IntVar()
        Checkbutton(master, text="dark_urine", variable=var34).grid(row=12,column=1, sticky=W)

        var35 = IntVar()
        Checkbutton(master, text="nausea", variable=var35).grid(row=12,column=2, sticky=W)

        var36 = IntVar()
        Checkbutton(master, text="loss_of_appetitie", variable=var36).grid(row=12,column=3, sticky=W)

        var37 = IntVar()
        Checkbutton(master, text="pain_behind_the_eyes", variable=var37).grid(row=13,column=0, sticky=W)

        var38 = IntVar()
        Checkbutton(master, text="back_pain", variable=var38).grid(row=13,column=1, sticky=W)

        var39 = IntVar()
        Checkbutton(master, text="constipation", variable=var39).grid(row=13, column=2,sticky=W)

        var40 = IntVar()
        Checkbutton(master, text="abdominal_pain", variable=var40).grid(row=13,column=3, sticky=W)

        var41 = IntVar()
        Checkbutton(master, text="diarrhoea", variable=var41).grid(row=14,column=0, sticky=W)

        var42 = IntVar()
        Checkbutton(master, text="mild_fever", variable=var42).grid(row=14,column=1, sticky=W)

        var43 = IntVar()
        Checkbutton(master, text="yellow_urine", variable=var43).grid(row=14,column=2, sticky=W)

        var44 = IntVar()
        Checkbutton(master, text="yellowing_of_eyes", variable=var44).grid(row=14, column=3,sticky=W)

        var45 = IntVar()
        Checkbutton(master, text="acute_liver_failure", variable=var45).grid(row=15,column=0, sticky=W)

        var46 = IntVar()
        Checkbutton(master, text="fluid_overload", variable=var46).grid(row=15,column=1, sticky=W)

        var47 = IntVar()
        Checkbutton(master, text="swelling_of_stomach", variable=var47).grid(row=15,column=2, sticky=W)

        var48 = IntVar()
        Checkbutton(master, text="swelled_lymph_nodes", variable=var48).grid(row=15,column=3, sticky=W)

        var49 = IntVar()
        Checkbutton(master, text="malaise", variable=var49).grid(row=16,column=0, sticky=W)

        var50 = IntVar()
        Checkbutton(master, text="blurred_and_distorted_vision", variable=var50).grid(row=16,column=1, sticky=W)

        var51 = IntVar()
        Checkbutton(master, text="phlegm", variable=var51).grid(row=16, column=2,sticky=W)

        var52 = IntVar()
        Checkbutton(master, text="throat_iritation", variable=var52).grid(row=16,column=3, sticky=W)

        var53 = IntVar()
        Checkbutton(master, text="redness_of_eyes", variable=var53).grid(row=17,column=0, sticky=W)

        var54 = IntVar()
        Checkbutton(master, text="sinus_pressure", variable=var54).grid(row=17,column=1, sticky=W)

        var55 = IntVar()
        Checkbutton(master, text="running_nose", variable=var55).grid(row=17, column=2,sticky=W)

        var56 = IntVar()
        Checkbutton(master, text="cogestion", variable=var56).grid(row=17,column=3, sticky=W)

        var57 = IntVar()
        Checkbutton(master, text="chest_pain", variable=var57).grid(row=18,column=0, sticky=W)

        var58 = IntVar()
        Checkbutton(master, text="weakness_in_limbs", variable=var58).grid(row=18,column=1, sticky=W)

        var59 = IntVar()
        Checkbutton(master, text="fast_heart_rate", variable=var59).grid(row=18, column=2,sticky=W)

        var60 = IntVar()
        Checkbutton(master, text="pain_during_bowel_movements", variable=var60).grid(row=18,column=3, sticky=W)

        var61 = IntVar()
        Checkbutton(master, text="pain_in_anal_region", variable=var61).grid(row=19,column=0, sticky=W)

        var62 = IntVar()
        Checkbutton(master, text="bloody_stool", variable=var62).grid(row=19,column=1, sticky=W)

        var63 = IntVar()
        Checkbutton(master, text="iritation_in_anus", variable=var63).grid(row=19, column=2,sticky=W)

        var64 = IntVar()
        Checkbutton(master, text="neck_pain", variable=var64).grid(row=19,column=3, sticky=W)

        var65 = IntVar()
        Checkbutton(master, text="dizziness", variable=var65).grid(row=20,column=0, sticky=W)

        var66 = IntVar()
        Checkbutton(master, text="cramps", variable=var66).grid(row=20,column=1, sticky=W)

        var67 = IntVar()
        Checkbutton(master, text="bruising", variable=var67).grid(row=20,column=2, sticky=W)

        var68 = IntVar()
        Checkbutton(master, text="obesity", variable=var68).grid(row=20,column=3, sticky=W)

        var69 = IntVar()
        Checkbutton(master, text="swollen_legs", variable=var69).grid(row=21, column=0,sticky=W)

        var70 = IntVar()
        Checkbutton(master, text="swollen_blood_vessels", variable=var70).grid(row=21,column=1, sticky=W)

        var71 = IntVar()
        Checkbutton(master, text="puffy_face_and_eyes", variable=var71).grid(row=21,column=2, sticky=W)

        var72 = IntVar()
        Checkbutton(master, text="enlarged_thyroid", variable=var72).grid(row=21,column=3, sticky=W)

        var73 = IntVar()
        Checkbutton(master, text="brittle_nails", variable=var73).grid(row=22,column=0, sticky=W)

        var74 = IntVar()
        Checkbutton(master, text="swellen_extremities", variable=var74).grid(row=22, column=1,sticky=W)

        var75 = IntVar()
        Checkbutton(master, text="excessive_hunger", variable=var75).grid(row=22,column=2, sticky=W)

        var76 = IntVar()
        Checkbutton(master, text="extra_marital_contacts", variable=var76).grid(row=22, column=3,sticky=W)

        var77 = IntVar()
        Checkbutton(master, text="drying_and_tingling_lips", variable=var77).grid(row=23,column=0, sticky=W)

        var78 = IntVar()
        Checkbutton(master, text="slurred_speech", variable=var78).grid(row=23,column=1, sticky=W)

        var79 = IntVar()
        Checkbutton(master, text="knee_pain", variable=var79).grid(row=23, column=2,sticky=W)

        var80 = IntVar()
        Checkbutton(master, text="hip_joint_pain", variable=var80).grid(row=23,column=3, sticky=W)

        var81 = IntVar()
        Checkbutton(master, text="muscle_weakness", variable=var81).grid(row=24,column=0, sticky=W)

        var82 = IntVar()
        Checkbutton(master, text="stiff_neck", variable=var82).grid(row=24,column=1, sticky=W)

        var83 = IntVar()
        Checkbutton(master, text="swelling_joints", variable=var83).grid(row=24,column=2, sticky=W)

        var84 = IntVar()
        Checkbutton(master, text="movements_stiffing", variable=var84).grid(row=24, column=3,sticky=W)

        var85 = IntVar()
        Checkbutton(master, text="spinning_movements", variable=var85).grid(row=25,column=0, sticky=W)

        var86 = IntVar()
        Checkbutton(master, text="loss_of_balance", variable=var86).grid(row=25,column=1, sticky=W)

        var87 = IntVar()
        Checkbutton(master, text="unsteadiness", variable=var87).grid(row=25, column=2,sticky=W)

        var88 = IntVar()
        Checkbutton(master, text="weakness_of_one_body_side", variable=var88).grid(row=25,column=3, sticky=W)

        var89 = IntVar()
        Checkbutton(master, text="loss_of_smell", variable=var89).grid(row=26,column=0, sticky=W)

        var90 = IntVar()
        Checkbutton(master, text="bladder_discomfort", variable=var90).grid(row=26,column=1, sticky=W)

        var91 = IntVar()
        Checkbutton(master, text="foul_smell_of_urine", variable=var91).grid(row=26,column=2, sticky=W)

        var92 = IntVar()
        Checkbutton(master, text="continuous_feel_of_urine", variable=var92).grid(row=26, column=3,sticky=W)

        var93 = IntVar()
        Checkbutton(master, text="passage_of_gas", variable=var93).grid(row=27,column=0, sticky=W)

        var94 = IntVar()
        Checkbutton(master, text="internal_itching", variable=var94).grid(row=27,column=1, sticky=W)

        var95 = IntVar()
        Checkbutton(master, text="toxic_look_(typhos)", variable=var95).grid(row=27,column=2, sticky=W)

        var96 = IntVar()
        Checkbutton(master, text="depression", variable=var96).grid(row=27,column=3, sticky=W)

        var97 = IntVar()
        Checkbutton(master, text="irritability", variable=var97).grid(row=28,column=0, sticky=W)

        var98 = IntVar()
        Checkbutton(master, text="muscle_pain", variable=var98).grid(row=28, column=1,sticky=W)

        var99 = IntVar()
        Checkbutton(master, text="altered_sensorium", variable=var99).grid(row=28,column=2, sticky=W)

        var100 = IntVar()
        Checkbutton(master, text="red_spots_over_body", variable=var100).grid(row=28,column=3, sticky=W)

        var101 = IntVar()
        Checkbutton(master, text="belly_pain", variable=var101).grid(row=29,column=0, sticky=W)

        var102 = IntVar()
        Checkbutton(master, text="abnormal_mensturation", variable=var102).grid(row=29, column=1,sticky=W)

        var103 = IntVar()
        Checkbutton(master, text="dischromic_patches", variable=var103).grid(row=29,column=2, sticky=W)

        var104 = IntVar()
        Checkbutton(master, text="watering_from_eyes", variable=var104).grid(row=29,column=3, sticky=W)

        var105 = IntVar()
        Checkbutton(master, text="increased_appetite", variable=var105).grid(row=30,column=0, sticky=W)

        var106 = IntVar()
        Checkbutton(master, text="polyuria", variable=var106).grid(row=30, column=1,sticky=W)

        var107 = IntVar()
        Checkbutton(master, text="family_history", variable=var107).grid(row=30,column=2, sticky=W)

        var108 = IntVar()
        Checkbutton(master, text="mucoid_sputum", variable=var108).grid(row=30,column=3, sticky=W)

        var109 = IntVar()
        Checkbutton(master, text="rusty_sputum", variable=var109).grid(row=31,column=0, sticky=W)

        var110 = IntVar()
        Checkbutton(master, text="lack_of_concentration", variable=var110).grid(row=31,column=1, sticky=W)

        var111 = IntVar()
        Checkbutton(master, text="visual_disturbance", variable=var111).grid(row=31,column=2, sticky=W)

        var112 = IntVar()
        Checkbutton(master, text="receiving_blood_transfusion", variable=var112).grid(row=31,column=3, sticky=W)

        var113 = IntVar()
        Checkbutton(master, text="receiving_unsterile_injections", variable=var113).grid(row=32, column=0,sticky=W)

        var114 = IntVar()
        Checkbutton(master, text="coma", variable=var114).grid(row=32,column=1, sticky=W)

        var115 = IntVar()
        Checkbutton(master, text="stomach_bleeding", variable=var115).grid(row=32,column=2, sticky=W)

        var116 = IntVar()
        Checkbutton(master, text="distention_of_abdomen", variable=var116).grid(row=32,column=3, sticky=W)

        var117 = IntVar()
        Checkbutton(master, text="history_of_alcohol_consumption", variable=var117).grid(row=33,column=0, sticky=W)

        var118 = IntVar()
        Checkbutton(master, text="fluid_overload", variable=var118).grid(row=33,column=1, sticky=W)

        var119 = IntVar()
        Checkbutton(master, text="blood_in_sputum", variable=var119).grid(row=33,column=2, sticky=W)

        var120 = IntVar()
        Checkbutton(master, text="prominent_veins_on_calf", variable=var120).grid(row=33,column=3, sticky=W)

        var121 = IntVar()
        Checkbutton(master, text="palpitations", variable=var121).grid(row=34,column=0, sticky=W)

        var122 = IntVar()
        Checkbutton(master, text="painful_walking", variable=var122).grid(row=34, column=1,sticky=W)

        var123 = IntVar()
        Checkbutton(master, text="pus_flilled_pimples", variable=var123).grid(row=34,column=2, sticky=W)

        var124 = IntVar()
        Checkbutton(master, text="blackheads", variable=var124).grid(row=34,column=3, sticky=W)

        var125 = IntVar()
        Checkbutton(master, text="scurring", variable=var125).grid(row=35,column=0, sticky=W)

        var126 = IntVar()
        Checkbutton(master, text="skin_peeling", variable=var126).grid(row=35,column=1, sticky=W)

        var127 = IntVar()
        Checkbutton(master, text="silver_like_dusting", variable=var127).grid(row=35,column=2, sticky=W)

        var128 = IntVar()
        Checkbutton(master, text="small_dents_in_nails", variable=var128).grid(row=35,column=3, sticky=W)

        var129 = IntVar()
        Checkbutton(master, text="inflammatory_nails", variable=var129).grid(row=36,column=0, sticky=W)

        var130 = IntVar()
        Checkbutton(master, text="blister", variable=var130).grid(row=36,column=1, sticky=W)

        var131 = IntVar()
        Checkbutton(master, text="red_sore_around_nose", variable=var131).grid(row=36, column=2,sticky=W)

        var132 = IntVar()
        Checkbutton(master, text="yellow_crust_ooze", variable=var132).grid(row=36, column=3,sticky=W)
        
        Button(master, text='QUIT', command=master.quit,fg="red").grid(row=16,column=19, sticky=W)
        Button(master, text='PREDICT', command=var_states,fg="green").grid(row=16, column=14,sticky=W)
        


    


        


              
    def new_user(self):
    	#Establish Connection
        with sqlite3.connect('quit.db') as db:
            c = db.cursor()

        #Find Existing username if any take proper action
        find_user = ('SELECT * FROM user WHERE username = ?')
        c.execute(find_user,[(self.username.get())])        
        if c.fetchall():
            ms.showerror('Error!','Username Taken Try a Diffrent One.')
        else:
            ms.showinfo('Success!','Account Created!')
            self.log()
        #Create New Account 
        insert = 'INSERT INTO user(username,password) VALUES(?,?)'
        c.execute(insert,[(self.n_username.get()),(self.n_password.get())])
        db.commit()

        #Frame Packing Methords
    def log(self):
        self.username.set('')
        self.password.set('')
        self.crf.pack_forget()
        self.head['text'] = 'LOGIN'
        self.logf.pack()
    def cr(self):
        self.n_username.set('')
        self.n_password.set('')
        self.logf.pack_forget()
        self.head['text'] = 'Create Account'
        self.crf.pack()
        
    #Draw Widgets
    def widgets(self):
        self.head = Label(self.master,text = 'LOGIN',font = ('',35),pady = 10)
        self.head.pack()
        self.logf = Frame(self.master,padx =10,pady = 10)
        Label(self.logf,text = 'Username: ',font = ('',20),pady=5,padx=5).grid(sticky = W)
        Entry(self.logf,textvariable = self.username,bd = 5,font = ('',15)).grid(row=0,column=1)
        Label(self.logf,text = 'Password: ',font = ('',20),pady=5,padx=5).grid(sticky = W)
        Entry(self.logf,textvariable = self.password,bd = 5,font = ('',15),show = '*').grid(row=1,column=1)
        Button(self.logf,text = ' Login ',bd = 3 ,font = ('',15),padx=5,pady=5,command=self.login,fg="green").grid(row=0,column=3)
        Button(self.logf,text = ' Create Account ',bd = 3 ,font = ('',15),padx=5,pady=5,command=self.cr,fg="blue").grid(row=0,column=2)
        Button(self.logf, text='PREDICT',bd=3,font = ('',15),padx=5,pady=5, command=self.visible,fg="olive").grid(row=1, column=3,sticky=W)
        Button(self.logf, text='RESET',bd=3,font = ('',15),padx=5,pady=5, command=self.reset,fg="red").grid(row=1, column=2,sticky=W)
        localtime=time.asctime(time.localtime(time.time()))
        self.lb=Label(self.logf,text = localtime,font=('arial',20,'bold' ),anchor='w')
        self.lb.grid(row=3,column=0)
        


        self.logf.pack()
        
        self.crf = Frame(self.master,padx =10,pady = 10)
        Label(self.crf,text = 'Username: ',font = ('',20),pady=5,padx=5).grid(sticky = W)
        Entry(self.crf,textvariable = self.n_username,bd = 5,font = ('',15)).grid(row=0,column=1)
        Label(self.crf,text = 'Password: ',font = ('',20),pady=5,padx=5).grid(sticky = W)
        Entry(self.crf,textvariable = self.n_password,bd = 5,font = ('',15),show = '*').grid(row=1,column=1)
        Button(self.crf,text = 'Create Account',bd = 3 ,font = ('',15),padx=5,pady=5,command=self.new_user,fg="blue").grid(row=0,column=2)
        Button(self.crf,text = 'Go to Login',bd = 3 ,font = ('',15),padx=5,pady=5,command=self.log,fg="green").grid(row=0,column=3)
        Button(self.crf, text='RESET',bd=3,font = ('',15),padx=5,pady=5, command=self.reset,fg="red").grid(row=1, column=2,sticky=W)

        Button(self.crf, text='PREDICT',bd=3,font = ('',15),padx=5,pady=5, command=self.visible,fg="olive").grid(row=1, column=3,sticky=W)
        localtime=time.asctime(time.localtime(time.time()))
        self.lb=Label(self.crf,text = localtime,font=('arial',20,'bold' ),anchor='w')
        self.lb.grid(row=3,column=0)

    def reset(self):
        
        self.n_username.set("")
        self.n_password.set("")
        self.username.set('')
        self.password.set('')
		
    

if __name__ == '__main__':
	#Create Object
	#and setup window
    root = Tk()
    root.title('Login Form')
    #root.geometry('400x350+300+300')
    main(root)
    root.mainloop()
