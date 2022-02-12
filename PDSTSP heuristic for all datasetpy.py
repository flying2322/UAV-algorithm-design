# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:41:33 2022
@ Weihui, Henan, China
@ author: wenpeng
@ PDSTSP heuristic: algorithm 6 and algorithm 7
@ solveTSP: Simulated annealing algorithm [1000, 20, 0.97, 100]
"""

import re
import math
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import datetime
#import pandas as pd


class Data:
    customerNum = 0
    nodeNum     = 0
    UAVrange    = 0
    cor_X       = []
    cor_Y       = []
    UAV_eligible_num = 0 
    disMatrix   = np.array([[]])
    
    
class Customer:
    idNum:  int
    x_cor:  float
    y_cor:  float
    withinRange: bool
 
    
path = 'In/r101.txt'
customerNum1 =100

Customers = [Customer() for i in range(customerNum1)]
    
# function to read data from .txt files
def readData(data, path, customerNum):
    data.customerNum = customerNum
    data.nodeNum = customerNum+2
    f = open(path, 'r')
    lines = f.readlines()
    count = 0
    countCus = 0
    # read the info to our variables
    for line in lines:
        count = count + 1
        if(count == 2):
            line = line[:-1]
            str = re.split(r" +", line)
            data.UAVrange = 40
        elif(count >= 10 and count <= 10 + customerNum):
            line = line[:-1]
            str = re.split(r" +", line)
            data.cor_X.append(float(str[2]))
            data.cor_Y.append(float(str[3]))
            if(count > 10 and count <= 10 + customerNum):      
                countCus = countCus +1
                Customers[countCus-1].idNum = int(str[1])
                Customers[countCus-1].x_cor = float(str[2])
                Customers[countCus-1].y_cor = float(str[3])
            
            
    data.cor_X.append(data.cor_X[0])
    data.cor_Y.append(data.cor_Y[0])
    
    # Compute the diatance matrix
    data.disMatrix = [([0] * data.nodeNum) for p in range(data.nodeNum)]
    # 初试化距离矩阵的维度，防止浅拷贝
    for i in range(0, data.nodeNum):
      for j in range(0, data.nodeNum):
          temp = (data.cor_X[i] - data.cor_X[j])**2 + (data.cor_Y[i] - data.cor_Y[j])**2
          data.disMatrix[i][j] = math.sqrt(temp)        
          temp = 0    

    return data


def printData(data,customerNum):
    for i in range(data.customerNum):   
        if(data.disMatrix[i+1][0] <= data.UAVrange):
            Customers[i].withinRange = 1  
        else:
            Customers[i].withinRange = 0
    for l in range(data.customerNum):
        if(Customers[l].withinRange == 1):
            data.UAV_eligible_num = data.UAV_eligible_num + 1
    print(" ***********Data Info***********\n")
    print(" UAV  range            = %4d" %data.UAVrange)
    print(" Customer's     number = %4d" %customerNum1)
    print(" UAV's eligible CusNum = %4d\n" %data.UAV_eligible_num)
    
    # print("*****************************Distance Matrix***************************")
    # for i in range(data.nodeNum):
    #     for j in range(data.nodeNum):
    #         print("%8.4f" %(data.disMatrix[i][j]), end  = " ")
    #     print()
    # print()  
        
#Reading data
data = Data()
uavSpeed = 2.0
truckSpeed = 1.0
readData(data, path, customerNum1) 
printData(data, customerNum1)

# plt.scatter(data.cor_X[0], data.cor_Y[0], c='blue', alpha=1, marker=',', linewidths=5, label='depot')
# plt.scatter(data.cor_X[1:-1], data.cor_Y[1:-1], c='magenta', alpha=1, marker='o', linewidths=5, label='customer') 

def solvePMS(cuss):
    # print("Function solvePMS() was called\n")
    uavMkspn        = 0
    tempuavAssignments  = []
    for i in range(len(cuss)):
        uavMkspn = uavMkspn + 2 * 0.5 * data.disMatrix[0][cuss[i].idNum]
        tempuavAssignments.append(cuss[i].idNum)
    uavAssignments = copy.deepcopy(tempuavAssignments)
    return [uavMkspn, uavAssignments]

    
def tsp_len(dis, path):#path format: <4 2 1 3 8 5 7 6 10 9>
    # dis: N*N adjcent matrix
    # verctor length is N 1
    N = len(path)
    leng = 0
    for i in range(N-1): # 0 1 2 3 ... 9
        leng = leng  +  dis[path[i]][path[i+1]]
    leng = leng + dis[0][path[N-1]]
    leng = leng + dis[0][path[0]]
    return leng    
  

def tsp_new_path(oldpath):
    #change oldpath to its neighbor
    N = len(oldpath)
    
    if(random.random() < 0.25): # generate two positions and change them
        chpos = random.sample(range(N),2)
        newpath = copy.deepcopy(oldpath)
        if(chpos[0] == chpos[1]):
            newpath = tsp_new_path(oldpath)
        newpath[chpos[1]] = oldpath[chpos[0]]
        newpath[chpos[0]] = oldpath[chpos[1]]
        
    else: # generate three place and change a-b & b-c
        d = random.sample(range(N),3)
        d.sort()
  
        a = d[0]
        b = d[1]
        c = d[2]

        if (a != b and b!=c):
            newpath = copy.deepcopy(oldpath)
            newpath[a:(c+1)]  = oldpath[b:(c+1)] + oldpath[a:b]
        else:
            newpath = tsp_new_path(oldpath)
    # print("Newpath:*********************")
    # print(newpath)
    return newpath                                                                        
        
    
def solveTSP(truckcuss):
    # print("Function solveTSP() was called\n")
    
    truckMkspn = 0
    truckRoute = []
    x0         = []
    
    for i in range(len(truckcuss)):
        x0.append(truckcuss[i].idNum)
   # x0.append() 
    
    dist        = data.disMatrix
    if(len(truckcuss)<3):
        # print('return function was called, which means the num of truck route=2')
        return [tsp_len(dist, x0), x0]
    
    MAX_ITER    = 1000
    MAX_M       = 20
    lambdaa     = 0.97
    T0          = 100 
    
    T       = T0
    ite     = 1
    x       = x0
    xx      = []
    xx.append(x) 
    di      = []
    di.append(tsp_len(dist, x0))
    n       = 1 
    
    while ite<MAX_ITER:
        m = 1
        while m<MAX_M:
            # generate new path()
            tempx   = []
            tempx   = tsp_new_path(x)
            newx    = copy.deepcopy(tempx)
            
            #calculate distance
            oldl = tsp_len(dist, x)
            newl = tsp_len(dist, newx)
            
            if(oldl > newl): # if new path is more superier, choose new path as the next status
                x           = copy.deepcopy(newx)
                xx.append(x)
                di.append(newl)
                n           = n + 1
            
            m = m + 1 
        ite = ite + 1
        T = T * lambdaa
        
    def indexofMin(arr):
        # print("Function indexofMin() was called\n")
        minindex = 0
        currentindex = 1
        while currentindex < len(arr):
            if arr[currentindex] < arr[minindex]:
                minindex = currentindex
            currentindex += 1
        return minindex
        
    truckMkspn = min(di)
    indexMin = indexofMin(di)
    
    # print("indexMin = %4d" %(indexMin))
    temptruckRoute = xx[indexMin]
    truckRoute = copy.deepcopy(temptruckRoute)
    return [truckMkspn, truckRoute]
            
    
def swap(umk, tmk, ua, tr): 
    # stand for: uavMkspn, truckMkspn, uavAssignments, truckRoute
    print("function SWAP was called, and uav Makespan, truck makespan, UAV customers and Truck customers' are: %.2f %.2f \n"%(umk,tmk) + str(ua) + str(tr))
       
    ms = 0 # maxSavings for return
    intersecCus = []
    for ii in range(len(tr)):
        if(Customers[tr[ii]-1].withinRange == 1):
            intersecCus.append(tr[ii])

         
    n1 = len(ua)
    n2 = len(intersecCus)
    

    
    
    
    for i in range(n1):
        for j in range(n2):
            tempuci       = ua[i]           # 备份无人机当前待交换顾客编号
            tempintersecj = intersecCus[j]  # 备份卡车带交换顾客编号
            
            backupua = copy.deepcopy(ua)
            ua.remove(tempuci)
            ua.append(intersecCus[j])
            uaP = copy.deepcopy(ua)
            
            # resotore the origin status of UAV service 
            ua  = copy.deepcopy(backupua)
            
            backuptr = copy.deepcopy(tr)
            tr.append(tempuci)
            tr.remove(tempintersecj)
            trP = copy.deepcopy(tr)
            
            # restore the origin status of TRUCK service
            tr = copy.deepcopy(backuptr)
            
            uavCusP  = []
            truckCusP = []
            for g in range(len(uaP)):
                uavCusP.append(Customers[uaP[g]-1])
            for w in range(len(trP)):
                truckCusP.append(Customers[trP[w]-1])           
            
            # Variable initialization
            umkP    = 0
            tmkP    = 0
            uasP    = []
            trP     = []
            
            [umkP, uasP]    = solvePMS(uavCusP)
            [tmkP, trP]     = solveTSP(truckCusP)
            
            objnew = max([umkP, tmkP])
            objold = max([umk, tmk])
            if(objold - objnew > ms):
                ms = objold - objnew
                umkPrime    = umkP
                tmkPrime    = tmkP
                
                uasPrime    = copy.deepcopy(uasP)
                trPrime     = copy.deepcopy(trP)
    
    if(ms>0):
        umk = umkPrime
        tmk = tmkPrime      
        ua  = copy.deepcopy(uasPrime)
        tr  = copy.deepcopy(trPrime)
        
        
    return [ms, umk, tmk, ua, tr]
            
        
    
    
def PDSTSPheuristic(allcus):
    #Initialize
    uavCustomers    = []
    truckCustomers  = []
    for i in range(len(allcus)):
        if(allcus[i].withinRange == 1):
            uavCustomers.append(allcus[i])
        else:
            truckCustomers.append(allcus[i])

    uavMkspn1       = 0
    truckMkspn1     = 0      
    uavAssignments1 = []
    truckRoute1     = []
    [uavMkspn1, uavAssignments1]  = solvePMS(uavCustomers)
    [truckMkspn1, truckRoute1]    = solveTSP(truckCustomers)
    
    

    while 1:
        if(uavMkspn1 > truckMkspn1):
             maxSavings     = 0
             backupUAVcus   = copy.deepcopy(uavAssignments1) # 备份无人机顾客的ID number
             countUAVCusNum = len(uavAssignments1)
             # iPrime         =0
             uavMkspnPrime  = 0
             truckMkspnPrime = 0

             for i in range(countUAVCusNum):
                 uavAssignmentsP    = [] # 定义无人机顾客顺序
                 truckRouteP        = [] # 定义卡车顾客服务顺序
                 uavAssignmentsPP   = [] # 
                 truckRoutePP       = [] #            
                 uavCustomersP      = [] # 无人机顾客
                 truckCustomersP    = [] # 卡车服务顾客
                 
                 # 依次添加<某个无人机访问顾客>到卡车路径中
                 tempuavA   = copy.deepcopy(uavAssignments1)
                 uavAssignments1.remove(backupUAVcus[i])
                 uavAssignmentsP    = copy.deepcopy(uavAssignments1)  
                 
                 temptrkA   = copy.deepcopy(truckRoute1)
                 truckRoute1.append(backupUAVcus[i])
                 truckRouteP        = copy.deepcopy(truckRoute1) #注意深度copy
                 
                 for g in range(len(uavAssignmentsP)):
                     uavCustomersP.append(Customers[uavAssignments1[g]-1])
                 for w in range(len(truckRouteP)):
                     truckCustomersP.append(Customers[truckRoute1[w]-1])
    
                 # 将调换后的顾客重新计算卡车和无人机最佳访问的最短路和顾客访问顺序
                 [uavMkspnP, uavAssignmentsPP] = solvePMS(uavCustomersP)
                 [truckMakespnP, truckRoutePP] = solveTSP(truckCustomersP)
                 
                 savings    =  uavMkspn1 - uavMkspnP
                 cost       =  truckMakespnP - truckMkspn1
                 
                 if((savings - cost) > maxSavings):
                     maxSavings = savings - cost
                     # iPrime     = i
                     # print("i = %4d" %i, end = " ") 
                     uavMkspnPrime      = uavMkspnP
                     truckMkspnPrime    = truckMakespnP
                     uavAssignmentsPrime    = copy.deepcopy(uavAssignmentsPP)
                     truckRoutePrime        = copy.deepcopy(truckRoutePP)
                 # 做完所有计算后，恢复成计算初的状态
                 uavAssignments1 = copy.deepcopy(tempuavA)
                 truckRoute1     = copy.deepcopy(temptrkA)
                 
             if(maxSavings > 0):
                print("maxSaving is : %6.2f" %maxSavings)
                uavAssignments1 = copy.deepcopy(uavAssignmentsPrime)
                truckRoute1     = copy.deepcopy(truckRoutePrime)
                uavMkspn1 = uavMkspnPrime
                truckMkspn1 = truckMkspnPrime
             else:
                [maxSavings, uavMkspn1, truckMkspn1, uavAssignments1,truckRoute1] = swap(uavMkspn1, truckMkspn1, uavAssignments1, truckRoute1)
                if(maxSavings == 0):
                    break
                
        else:
            [maxSavings, uavMkspn1, truckMkspn1, uavAssignments1,truckRoute1] = swap(uavMkspn1, truckMkspn1, uavAssignments1, truckRoute1)
            if(maxSavings == 0):
                break
        
    print("PDSTSP heuristic (Algorithm6) was successfully called!")      
    return  [uavMkspn1, truckMkspn1, uavAssignments1, truckRoute1]
            
  

if __name__ == "__main__":
    uavMakespn      = 0
    truckMakespn    = 0
    uavAssign       = []
    truckRoute      = []
    starttime = datetime.datetime.now()
    [uavMakespn, truckMakespn, uavAssign, truckRoute] = PDSTSPheuristic(Customers)
    endtime = datetime.datetime.now()
    print('\n*************** The optimal solution are AS FOLLOWS: *************\n')
    print('UAV makespan  : %5.2f' %(uavMakespn))
    print('Truck makespan: %5.2f' %(truckMakespn))
    print('UAV Assignments  : ' + str(uavAssign))
    print('Truck Assignments: ' + str(truckRoute))

    strrr="run time: %d seconds" % ((endtime - starttime).seconds)
    print(strrr)

    print('\n******* Detailed path info was shown in PLOTS windows above! *****')
    
    # draw the route graph
    # draw all the nodes first
    # data1 = Data() 
    # readData(data1, path, 100) 
    fig = plt.figure(figsize=(15,10)) 
    font_dict = {'family': 'Arial',   # serif
         'style': 'normal',   # 'italic',
         'weight': 'normal',
        'color':  'darkred', 
        'size': 30,
        }
    font_dict2 = {'family': 'Arial',   # serif
         'style': 'normal',   # 'italQic',
         'weight': 'normal',
        'color':  'darkred', 
        'size': 24,
        }
    plt.xlabel('x', font_dict) 
    plt.ylabel('y', font_dict)
    plt.title('Optimal Solution for PDSTSP heuristic', font_dict)  
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)    # plt.yticks(fontsize=30) 
    plt.grid(True, color='r', linestyle='-', linewidth=2)
    
    
    '''
    marker='o'
    marker=','
    marker='.'
    marker=(9, 3, 30)
    marker='+'
    marker='v'
    marker='^'
    marker='<'
    marker='>'
    marker='1'
    marker='2'
    marker='3'
    red        blue        green
    '''
    plt.scatter(data.cor_X[0], data.cor_Y[0], c='blue', alpha=1, marker=',', linewidths=5, label='depot')
    plt.scatter(data.cor_X[1:-1], data.cor_Y[1:-1], c='magenta', alpha=1, marker='o', linewidths=5, label='customer') 



    # Drew the route
    lengthTR = len(truckRoute)
    for i in range(lengthTR-1):
        x = [Customers[truckRoute[i]-1].x_cor, Customers[truckRoute[i+1]-1].x_cor]
        y = [Customers[truckRoute[i]-1].y_cor, Customers[truckRoute[i+1]-1].y_cor]
        plt.plot(x, y, 'b', linewidth = 3)
        plt.text(Customers[truckRoute[i]-1].x_cor-0.2, Customers[truckRoute[i]-1].y_cor, str(truckRoute[i]), fontdict  = font_dict2)
        
    # conect depot to the first customer
    # x = [data.cor_X[0], Customers[truckRoute[0]-1].x_cor]
    # y = [data.cor_Y[0], Customers[truckRoute[0]-1].y_cor]    
    x = [data.cor_X[0], data.cor_X[truckRoute[0]]]
    y = [data.cor_Y[0], data.cor_Y[truckRoute[0]]]  
    plt.plot(x, y, 'b', linewidth = 3)
    plt.text(data.cor_X[truckRoute[0]]-0.2, data.cor_Y[truckRoute[0]], str(truckRoute[0]), fontdict  = font_dict2)
    
    # conect depot to the last customer
    x = [data.cor_X[0], data.cor_X[truckRoute[lengthTR-1]]]
    y = [data.cor_Y[0], data.cor_Y[truckRoute[lengthTR-1]]]    
    plt.plot(x, y, 'b', linewidth = 3)
    plt.text(data.cor_X[truckRoute[lengthTR-1]]-0.2, data.cor_Y[truckRoute[lengthTR-1]], str(truckRoute[lengthTR-1]), fontdict  = font_dict2)
    
    
    
    for i in range(len(uavAssign)):
        x = [data.cor_X[0], data.cor_X[uavAssign[i]]]
        y = [data.cor_Y[0], data.cor_Y[uavAssign[i]]]
        plt.plot(x, y, 'r--', linewidth = 3)
        plt.text(data.cor_X[uavAssign[i]]-0.2, data.cor_Y[uavAssign[i]], str(uavAssign[i]), fontdict=font_dict2)
        
    #plt.grid(True)
    plt.grid(False)
    plt.legend(loc='best', fontsize = 20)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
