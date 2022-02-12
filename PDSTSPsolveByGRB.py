# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:05:25 2022
Target: solve PDSTSP by call GRB
@author: wenpeng
"""
from __future__ import print_function
from gurobipy import *

import re
import math
import matplotlib.pyplot as plt
# import random
import numpy as np
# import copy
import datetime
import pandas as pd


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
 
#################################    
path = 'c101.txt'               #
customerNum1    = 9            #
UAVnum          = 1             #
#################################


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
    
    print("*****************************Distance Matrix***************************")
    for i in range(data.nodeNum):
        for j in range(data.nodeNum):
            print("%5.2f" %(data.disMatrix[i][j]), end  = " ")
        print()
    print()  
 

class Solution:
    ObjVal = 0
    X = [[]]
    Y = [[]]
    U = []
    # z:float
    
    route_Truck = []
    route_UAV   = [[]]
    
    def getSolution(self, data, model):
        solution = Solution()
        solution.ObjVal = model.ObjVal
        solution.X = [([0] * data.nodeNum) for j in range(data.nodeNum)]
        solution.Y = [([0] * UAVnum) for v in range(customerNum1)]
        solution.U = [[0] for i in range(data.nodeNum)]
        solution.z = 0
        
        # a = U[0].x
        for m in model.getVars():
            str = re.split(r"_", m.varName)
            if(str[0] == "X" and m.x ==1):
                solution.X[int(str[1])][int(str[2])] = m.x
                print(str, end="")
                print(" = %d" %m.x)
            elif(str[0] == 'Y' and m.x == 1):
                solution.Y[int(str[2])] = m.x
            elif(str[0] == 'U' and m.x > 0):
                solution.U[int(str[1])] = m.x
            elif(str[0] == 'z' and m.x >0):
                solution.z = m.x
        # get the route of truck and UAV
        j = 0
        for i in range(data.nodeNum):
           i = j # note that the variable is whether is alocal variable or a globle variable
           for j in range(data.nodeNum):
               if(solution.X[i][j] == 1):
                   solution.route_Truck.append(i)
                   print(" %d -" % i, end = " ")
                   break
        print(" 0")
        solution.route_Truck.append(0)
        
        print("\n\n --------- Route of UAV ---------")
        # count = 0
        for v in range(UAVnum):
            solution.route_UAV.append([0])
            print(" 0 ", end = "")
            for i in range(customerNum1):
                if(solution.Y[i] == 1):
                    solution.route_UAV[v].append(i+1)
                    print("- %d -" %(i+1), end = " ")
                    print("0 ", end = " ")
            
        return solution   



        
#Reading data
data = Data()
# uavSpeed = 2.0
# truckSpeed = 1.0
readData(data, path, customerNum1) 
printData(data, customerNum1)




# =========================Build the model=======================
big_M = 10000
# construct the model object
model = Model("PDSTSP_by_Gurobi")

# Initialize variables
# create variables: Muiti-dimension vector: from inner to outer
# X_ij hat
X = [[[] for i in range(data.nodeNum)] for j in range(data.nodeNum)]

# Y_iv hat
Y = [[[] for v in range(customerNum1+1)] for i in range(UAVnum)]

# U_i
U = [[] for i in range(data.nodeNum)]

# z
z: float


for i in range(data.nodeNum):
    name1 = 'U_' + str(i)
    U[i] = model.addVar(0, data.nodeNum, vtype = GRB.CONTINUOUS, name = name1)
    for j in range(data.nodeNum):
        name2 = 'X_' + str(i) + "_" + str(j)
        X[i][j] = model.addVar(0, 1, vtype = GRB.BINARY, name = name2)
# for i in range(1, customerNum1+1): 
for v in range(UAVnum):
    for k in range(customerNum1):
        name3 = 'Y_' + str(v) + '_' + str(k)
        Y[v][k] = model.addVar(0, 1, vtype = GRB.BINARY, name = name3)

z=model.addVar(0, big_M, vtype = GRB.CONTINUOUS, name = 'z')


# Add contraints        
# create the objective expressive(1)
obj = LinExpr(0)

# add thee objective funtion into the model
model.setObjective(z, GRB.MINIMIZE)


# constraint (1)
expr = LinExpr(0)
for i in range(data.nodeNum-1):
    for j in range(1, data.nodeNum):
        if (i != j):
            expr.addTerms(data.disMatrix[i][j], X[i][j])
model.addConstr(expr <= z, 'c1')
expr.clear()

# constraint (2)
expr = LinExpr(0)
for v in range(UAVnum):
    expr = LinExpr(0)
    expr.addTerms(data.disMatrix[0][0], Y[v][0])
    for i in range(1, customerNum1+1):
        expr.addTerms(data.disMatrix[0][i], Y[v][i-1])
    model.addConstr(expr <= z , 'c2')
    expr.clear()

# constrait (3)
expr = LinExpr(0)
for j in range(1,data.nodeNum-1):
    for i in range(data.nodeNum-1):
        if(i!=j):
            expr.addTerms(1, X[i][j])
    for v in range(UAVnum):
        if(Customers[j-1].withinRange == 1):
            expr.addTerms(1, Y[v][j-1])
    model.addConstr(expr == 1, 'c3')
    expr.clear()

# constraint (4)
expr = LinExpr(0)
for j in range(1, data.nodeNum):
    expr.addTerms(1, X[0][j])
model.addConstr(expr == 1, 'c4')#其中包括了depot到depot的弧
expr.clear()

#constraint (5)
expr = LinExpr(0)
for i in range(data.nodeNum-1):
    expr.addTerms(1, X[i][customerNum1+1])
model.addConstr(expr == 1, 'c5')
expr.clear()

#constraint (6)
for j in range(1, customerNum1+1):
    expr1 = LinExpr(0)
    expr2 = LinExpr(0)
    for i in range(customerNum1+1):
        if(i!=j):
            expr1.addTerms(1, X[i][j])
    for k in range(1, data.nodeNum):
        if(k!=j):
            expr2.addTerms(1, X[j][k])
    model.addConstr(expr1 == expr2, 'c6')
    expr1.clear()
    expr2.clear()
    
# constraint (7)   

for i in range(1, customerNum1+1):
    for j in range(1, customerNum1+2):
        if(i!=j):
            model.addConstr(U[i]-U[j]+1<=(data.nodeNum)*(1-X[i][j]),'c7')
            

            
# solve the problem
model.write('b.lp')
model.Params.timelimit = 7200*6
model.optimize()

# get the solution info
solution1 = Solution()
solution = solution1.getSolution(data, model)
print("\n\n\n-----optimal value-----")
print("Obj: %g" % solution.ObjVal) 
print("\n\n ------Route of Truck------")
j = 0
for i in range(data.nodeNum):
    i = j
    for j in range(data.nodeNum):
        if(solution.X[i][j] == 1):
            print(" %d -" % i, end = " ")
            break
print(" 0")


print("\n\n -------- Route of UAV --------")
for v in range(UAVnum):
    print("UAV-%d :"%v, end = " ")
    for j in range(customerNum1):
        if(solution.Y[j] == 1):
            print(" %d -" %(j+1), end = " ")
    print()

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
plt.title('Optimal Solution for PDSTSP  by Gurobi', font_dict)  
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


# Draw the route
for i in range(data.nodeNum):
    for j in range(data.nodeNum):
        if(solution.X[i][j] == 1):
            x = [data.cor_X[i], data.cor_X[j]]
            y = [data.cor_Y[i], data.cor_Y[j]]
            plt.plot(x, y, 'b', linewidth = 3)
            plt.text(data.cor_X[i]-0.2, data.cor_Y[i], str(i), fontdict = font_dict2)
            
            
for v in range(UAVnum):
    for j in range(customerNum1):
        if(solution.Y[j] == 1):
            x = [data.cor_X[0], data.cor_X[j+1]]
            y = [data.cor_Y[0], data.cor_Y[j+1]]
            plt.plot(x, y, 'r--', linewidth = 3)
            plt.text(data.cor_X[j+1]-0.2, data.cor_Y[j+1], str(j+1), fontdict = font_dict2)
            
            
# plt.frid(True)
plt.grid(False)
plt.legend(loc='best', fontsize = 20)
plt.show()    













































