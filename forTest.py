# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:53:29 2022

@author: wenpeng
"""
import datetime
import time


class Cus():
    def __init__(self, idd, cor_x, cor_y):
        self.idd = idd
        self.cor_x = cor_x
        self.cor_y = cor_y
    idd:     int
    cor_x:  float
    cor_y:  float
    
    
def TestClassAsPara(a):
    for i in range(len(a)):
        print(a[i].idd)
        print(a[i].cor_x)
        print(a[i].cor_y)
    
cusss1 = Cus(12, 34, 45)
cusss2 = Cus(15, 39, 40)

cuss=[]
cuss.append(cusss1)
cuss.append(cusss2)


hh: int



if __name__ == "__main__":
    
    starttime = datetime.datetime.now()

    #long running
    time.sleep(2)
    
    endtime = datetime.datetime.now()
    
    str="run time: %d seconds" % ((endtime - starttime).seconds)
    print(str)
    
    
    # begin   = time.clock()
    # TestClassAsPara(cuss)
    # endtime = time.clock()
    # print('main program started!')
    # print (endtime - begin)