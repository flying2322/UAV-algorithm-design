# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:41:33 2022
@weihui home, Henan, China
@author: wenpeng
"""

import re
import math
import matplotlib.pyplot as plt
import random
import numpy as np
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
 
    
path = 'c101.txt'
customerNum1 = 10

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
            data.UAVrange = float(str[0])
        elif(count >= 9 and count <= 9 + customerNum):
            line = line[:-1]
            str = re.split(r" +", line)
            data.cor_X.append(float(str[2]))
            data.cor_Y.append(float(str[3]))
            if(count > 9 and count <= 9 + customerNum):      
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
        if(data.disMatrix[i][0] <= data.UAVrange):
            Customers[i].withinRange = 1  
        else:
            Customers[i].withinRange = 0
    for l in range(data.customerNum):
        if(Customers[l].withinRange == 1):
            data.UAV_eligible_num = data.UAV_eligible_num + 1
    print(" ***********Data Info***********\n")
    print(" UAV range = %4d" %data.UAVrange)
    print(" UAV's eligible CusNum = %4d\n\n" %data.UAV_eligible_num)
    
    print("*********Distance Matrix***********")
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            print("%6.2f" %(data.disMatrix[i][j]), end  = " ")
        print()
      
        
#Reading data
data = Data()
uavSpeed = 2.0
truckSpeed = 1.0
readData(data, path, customerNum1) 
printData(data, customerNum1)

plt.scatter(data.cor_X[0], data.cor_Y[0], c='blue', alpha=1, marker=',', linewidths=5, label='depot')
plt.scatter(data.cor_X[1:-1], data.cor_Y[1:-1], c='magenta', alpha=1, marker='o', linewidths=5, label='customer') 

def solvePMS(cuss):
    print("Function solvePMS() was called\n")
    uavMkspn        = 0
    uavAssignments  = []
    for i in range(len(cuss)):
        uavMkspn = uavMkspn + 2 * 0.5 * data.disMatrix[0][cuss[i].idNum]
        uavAssignments.append(cuss[i].idNum)
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
        chpos = [math.floor(random.random()*N), math.floor(random.random()*N)]
        newpath = oldpath
        newpath[chpos[1]] = oldpath[chpos[0]]
        newpath[chpos[0]] = oldpath[chpos[1]]
        
    else: # generate three place and change a-b & b-c
        d = [math.ceil(random.random()*N),math.floor(random.random()*N),math.floor(random.random()*N)]
        d.sort()
        a = d[0]
        b = d[1]
        c = d[2]
        
        if (a != b and b!=c):
            newpath = oldpath
            newpath[a+1:c-1]  = oldpath[b:c-1] + oldpath[a+1:b-1]
        else:
            newpath = tsp_new_path(oldpath)        
    return newpath                                                                        
        
    
def solveTSP(truckcuss):
    print("Function solveTSP() was called\n")
    
    truckMkspn = 0
    truckRoute = []
    x0 = []
    
    for i in range(len(truckcuss)):
        x0.append(truckcuss[i].idNum)
   # x0.append() 
    
    dist        = data.disMatrix
    if(len(truckcuss)<3):
        print('return function was called')
        return [tsp_len(dist, x0), x0]
    MAX_ITER    = 2000
    MAX_M       = 20
    lambdaa     = 0.97
    T0          = 100 
    
    T       = T0
    ite     = 1
    x       = x0
    xx      = [[]]
    xx[0:]  = x 
    di      = []
    di.append(tsp_len(dist, x0))
    n       = 1 
    
    while ite<MAX_ITER:
        m = 1
        while m<MAX_M:
            # generate new path()
            newx = tsp_new_path(x)
            
            #calculate distance
            oldl = tsp_len(dist, x)
            newl = tsp_len(dist, newx)
            
            if(oldl > newl): # if new path is more superier, choose new path as the next status
                x           = newx
                xx[n+1:]    = x
                di[n+1]     = newl
                n           = n + 1
            
            m = m + 1 
        ite = ite + 1
        T = T * lambdaa
        
    def indexofMin(arr):
        print("Function indexofMin() was called\n")
        minindex = 0
        currentindex = 1
        while currentindex < len(arr):
            if arr[currentindex] < arr[minindex]:
                minindex = currentindex
            currentindex += 1
        return minindex
        
    truckMkspn = min(di)
    indexMin = indexofMin(di)
    
    print("indexMin = %4d" %(indexMin))
    truckRoute = xx[indexMin:]
    return [truckMkspn, truckRoute]
            
    
def swap(umk, tmk, uc, tc): 
    # stand for: uavMkspn, truckMkspn, uavCustomers, truckCustomers, cpp
    print("function SWAP was called, and uavMakespan truck makespan UAV customers and Truck customers' are: \n")
    print(umk)
    print(tmk)
    for i in range(len(uc)):
        print("%4d" %(uc[i].idNum), end=" ")
    print()
    for i in range(len(tc)):
        print("%4d" %(tc[i].idNum), end=" ")
       
    ms = 0 # maxSavings 
    intersecCus = []
    for ii in range(len(tc)):
        if(Customers[ii-1].withinRange == 1):
            intersecCus.append(Customers[ii-1])
    if not intersecCus:
        print("if not was called\n")
        umk1    = 0
        tmk1    = 0
        uas1    = []
        tr1     = []
        [umk1, uas1]    = solvePMS(uc)
        [tmk1, tr1]     = solveTSP(tc)
        return [0, umk1, tmk1, uas1, tr1]         
    n1 = len(uc)
    n2 = len(intersecCus)
    
    
    # print("now: UAV customers and intersection customers are: ")
    # for i in range(len(uc)):
    #     print("%4d" %(uc[i].idNum))
    # print()
    # for i in range(len(intersecCus)):
    #     print("%4d" %(intersecCus[i].idNum))
    
    
    
    for i in range(n1):
        for j in range(n2):
            tempuci = uc[i]
            tempintersecj = intersecCus[j]
            
            uc.remove(uc[i])
            uc.append(intersecCus[j])
            ucP = uc
            # resotore the origin status
            uc.append(tempuci)
            uc.remove(tempintersecj)
                    
            tc.append(tempuci)
            tc.remove(tempintersecj)
            tcP = tc
            
            tc.remove(tempuci)
            tc.append(tempintersecj)
            
            [umkP, uasP]    = solvePMS(ucP)
            [tmkP, trP]     = solveTSP(tcP)
            if((max[umk, tmk] - max[umkP, tmkP] > ms)):
                ms = max[umk, tmk] - max[umkP, tmkP]
                umkPrime    = umkP
                tmkPrime    = tmkP
                
                uasPrime    = uasP
                trPrime     = trP
    if(ms>0):
        umk = umkPrime
        tmk = tmkPrime
        
        uas = uasPrime
        tr  = trPrime
        
    return [ms, umk, tmk, uas, tr]
            
        
    
    
def PDSTSPheuristic(allcus):
    #Initialize
    uavCustomers    = []
    truckCustomers  = []
    for i in range(len(allcus)):
        if(allcus[i].withinRange == 1):
            uavCustomers.append(allcus[i])
        else:
            truckCustomers.append(allcus[i])
 #   cpp             = uavCustomers
    uavMkspn1       = 0
    truckMkspn1     = 0      
    uavAssignments1 = []
    truckRoute1     = []
    [uavMkspn1, uavAssignments1]  = solvePMS(uavCustomers)
    [truckMkspn1, truckRoute1]    = solveTSP(truckCustomers)
    
    print("*****Init: uavMkspn = %6.2f, and the uav's assignment is" %(uavMkspn1))
    print(uavAssignments1)
    print("\n*****Init: truckMkspn = %6.2f, and the truck's assignment is\n" %(truckMkspn1))
    print(truckRoute1)
    
    while 1:
        if(uavMkspn1 > truckMkspn1):
            
            # seek to improve solution by reducing the UAV makespan
            maxSavings = 0

            for i in range(len(uavCustomers)): # assign a uavCustomer to Truck's route
                print("Before truckCustomers = ")
                for k in range(len(truckCustomers)):
                    print(truckCustomers[k].idNum)    
                
                truckCustomers.append(uavCustomers[i])
                truckCustomersP = truckCustomers
                
                print("After truckCustomers = ")
                for k in range(len(truckCustomers)):
                    print(truckCustomers[k].idNum)
                truckCustomers.remove(uavCustomers[i])
                print("After remove, truckCustomers = ")
                for k in range(len(truckCustomers)):
                    print(truckCustomers[k].idNum)
                
                temporigin = uavCustomers[i]
                print("Now i = %d\n"  %i)
                uavCustomers.remove(uavCustomers[i])
                uavCustomersP   = uavCustomers
                uavCustomers.append(temporigin)

                
                [uavMkspnP, uavAssignmentsP]    = solvePMS(uavCustomersP)
                [truckMkspnP, truckRouteP]      = solveTSP(truckCustomersP)
                


                print("*****Now: uavMkspn = %6.2f, and the uav's assignment is" %(uavMkspnP))
                print(uavAssignmentsP)
                print("\n*****Now: truckMkspn = %6.2f, and the truck's assignment is" %(truckMkspnP))
                print(truckRouteP)
    
                savings = uavMkspn1 - uavMkspnP
                cost    = truckMkspn1 - truckMkspnP
                
                if((savings - cost) > maxSavings ):

                    maxSavings          = savings - cost
                    iPrime              = i
                    uavMkspnPrime       = uavMkspnP
                    # uavMkspn1           = uavMkspnPrime
                    truckMkspnPrime     = truckMkspnP
                    # truckMkspn1         = truckMkspnPrime
                    uavAssignmentsPrime = uavAssignmentsP
                    truckRoutePrime     = truckRouteP
            # Find the optimal i to min the 
            if(maxSavings > 0):
                truckCustomers  = truckCustomers.append(iPrime)
                uavCustomers    = uavCustomers.remove(iPrime)
                
                uavMkspn1        = uavMkspnPrime
                truckMkspn1      = truckMkspnPrime
            else:
                [maxSavings, uavMkspn1, truckMkspn1, uavAssignments1, truckRoute1] = swap(uavMkspn1, truckMkspn1, uavCustomers,truckCustomers)
                if(maxSavings == 0):
                    break # stop, no improved solution found via the swap
                
        else:
            [maxSavings, uavMkspn1, truckMkspn1, uavAssignments1, truckRoute1] = swap(uavMkspn1, truckMkspn1, uavCustomers,truckCustomers)
            if(maxSavings == 0):
                break # stop, no improved solution found via the swap
        
    return  [uavMkspn1, truckMkspn1, uavAssignments1, truckRoute1]
            
  

if __name__ == "__main__":
    uavMakespn      = 0
    truckMakespn    = 0
    uavAssign       = []
    truckAssign     = []
    [uavMakespn, truckMakespn, uavAssign, truckAssign] = PDSTSPheuristic(Customers)
    print('UAV makespan: %6.2f' %(uavMakespn))
    print('Truck makespan: %6.2f' %(truckMakespn))
    print('UAV Assignments:')
    print(uavAssign)
    print('Truck Assignments:')
    print(truckAssign)
    print('*********** ->PDSTSP heuristic algorithm<- ***********')
