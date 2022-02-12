​
本文复现的是论文【1】的第二部分PDSTSP问题的求解：The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery - 百度学术 (baidu.com)

 本文的代码复现参考了师兄的这篇帖子：

论文代码复现 | 无人机与卡车联合配送(Python+Gurobi)(The flying sidekick traveling salesman problem)_HsinglukLiu的博客-CSDN博客

因为在本论文中作者提出了两个模型，一是有无人机辅助的旅行商（The flying sidekick traveling salesman problem）问题（简记：FSTSP），二是并行无人机的调度优化问题（Parallel drone scheduling TSP, 简记为：PDSTSP）。这两类问题都相当于利用无人机的灵活配送效率高的优势进行建模，论文构建了两类问题的数学规划模型。

简单理解第一种建模FSTSP问题是将无人机和卡车在配送过程中进行协同，如下图【Fig.3】所示，无人机是从卡车在配送过程中的顾客点进行起降，同时在论文【1】中，为了简化问题，无人机路径只能和卡车路径组成三角形，即在卡车连续服务的两个顾客间无人机必须完成顾客服务的往返,例如图中的【4-7-6】和【6-1-5】。



 第二种建模方式PDSTSP问题是指无人机和卡车彼此并行作业，当任务分配完成后，即不再有无人机和卡车的交互,见下图【Fig.2】。例如在图Fig.2（a）中的路径时间明显大于图b和图c，该问题相当于在初始给定一定量的顾客后我们进行两个决策，第一是哪些顾客交由无人机进行服务，哪些顾客交由卡车服务，第二部，在给定的顾客分配下，确定无人机和卡车的最优服务的顺序和路径。本文正是复现的PDSTSP问题的代码。



 本文复现了PDSTSP三部分的代码，首先是直接调用求解器Gurobi求解论文的数学规划模型：



代码实现：

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
第二是对论文中提出的PDSTSP启发式伪代码进行Python实现，第三部分的工作是比较不同TSP求解方法对PDSTSP启发式方法求解质量和效率的影响。其中我们谈到的PDSTSP启发式方法是指下面的伪代码【包含Alorithm6和Algorithm7】：







简单理解PDSTSP heuristic的算法6和算法7如下：首先从已知的所有顾客中分辩出可以由无人机进行服务的顾客（主要指在无人机的电池耐力范围内的顾客）和只能由卡车进行服务的顾客（顾客的需求重量超出无人机的服务载荷，或需要顾客进行签字或者当面操作的情况），在初始化时将所有无人机可以服务的顾客全部分派给无人机，剩余不能由无人机服务的顾客分派给卡车，这样分派给无人机的顾客用求解PMS的算法进行安排最优服务顺序和路径（记为问题A，后面再展开），分派给卡车的顾客调用求解TSP问题的算法求解（记为问题B，同A，后面展开），通过比较问题A和B的服务时间，最大者即为原PDSTSP问题的解。

当然这样的初始解通常都有很大的优化空间：例如在论文【1】中，作者设置了不同比例80%，90%的顾客可以由无人机服务，这就意味着服务的主要服务载荷的loading在无人机，意味着无人机服务的时长通常会大于卡车的服务时长，这样进行第一步PDSTSP启发式的优化操作：将在无人机上的被服务的顾客安排到卡车上，那么选择哪一个顾客做这样的移动呢？答案是进行遍历所有无人机上的顾客，放到卡车上之后使原问题（PDSTSP问题本身）目标函数值最小（也就是最优）的移动方案，这样就完成了一次移动，重复这样的过程，无人机上的顾客数量在逐渐减少，卡车的服务顾客在逐渐增加，这样原来由无人机主导的服务时间会逐渐降低，卡车的服务时间会逐渐增加，最后无人机上顾客服务时间会在某一位顾客从无人机上移动到卡车上之后，卡车的服务时间大于无人机的服务时间，这时第一阶段的优化结束。

进行第二阶段PDSTSP的优化，即算法7: 将现在在卡车上的能被无人机服务的顾客再和现在在无人机上服务的顾客进行交换，如果依然可以实现当前的目标函数值的优化，就继续进行，直到没有可以继续优化的空间，目标函数值保持不变，算法结束。

在这部分的实现中，有两个点没有展开，就是问题A和B的求解：这两个问题也是运筹学中的经典问题，第一个是Parallel Machine Scheduling (PMS) problem给定无人机服务顾客进行任务调度优化的问题，论文中分享了两篇求解的论文，有兴趣的同学可以查阅参考文献【2】【3】，这里简单讲一下只有一架无人机的场景，相当于PMS问题的机台只有一台，那问题就退化为所有客户提供服务的时间总和即为问题的目标函数值。并且顾客的先后服务顺序不会影响最后的PMS的目标函数值。第二个是求解Traveling Salesman Problem（TSP），这个运筹学中的经典问题可以用多种方法来求解，本文分别使用Gurobi和模拟退火（Simulated Annealing）算法进行了尝试。代码如下：

Gurobi解TSP的PDSTSP启发式实现：

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:41:33 2022
@ Weihui, Henan, China
@ author: wenpeng
@ PDSTSP heuristic: algorithm 6 and algorithm 7
@ solveTSP: Called Gurobi to solve IP formaulation-TSP
"""
from gurobipy import *
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
 
    
path = 'c101.txt'
customerNum1 = 5

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
    
    # print("*****************************Distance Matrix***************************")
    # for i in range(data.nodeNum):
    #     for j in range(data.nodeNum):
    #         print("%5.2f" %(data.disMatrix[i][j]), end  = " ")
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

    
 
  

# def tsp_new_path(oldpath):
#     #change oldpath to its neighbor
#     N = len(oldpath)
    
#     if(random.random() < 0.25): # generate two positions and change them
#         chpos = random.sample(range(N),2)
#         newpath = copy.deepcopy(oldpath)
#         if(chpos[0] == chpos[1]):
#             newpath = tsp_new_path(oldpath)
#         newpath[chpos[1]] = oldpath[chpos[0]]
#         newpath[chpos[0]] = oldpath[chpos[1]]
        
#     else: # generate three place and change a-b & b-c
#         d = random.sample(range(N),3)
#         d.sort()
  
#         a = d[0]
#         b = d[1]
#         c = d[2]

#         if (a != b and b!=c):
#             newpath = copy.deepcopy(oldpath)
#             newpath[a:(c+1)]  = oldpath[b:(c+1)] + oldpath[a:b]
#         else:
#             newpath = tsp_new_path(oldpath)
#     # print("Newpath:*********************")
#     # print(newpath)
#     return newpath                                                                        
def tsp_len(dis, path):#path format: < 8 5 7 6 >
# dis: N*N adjcent matrix
# verctor length is N 1

    
    NN1 = len(path)
    leng = 0
    if(NN1 == 1):
        leng = 2*dis[0][1]
    else:
        for i in range(NN1-1): # 0 1 2 3 ... 9
            leng = leng  +  data.disMatrix[path[i]][path[i+1]]
        leng = leng + data.disMatrix[0][path[NN1-1]]
        leng = leng + data.disMatrix[0][path[0]]
    return leng    



     
def getValue(var_dict, nodeNumm):
    x_value = np.zeros([nodeNumm + 1, nodeNumm + 1])
    for key in var_dict.keys():
        a = key[0]
        b = key[1]
        x_value[a][b] = var_dict[key].x
        
    return x_value




def getRoute(x_value):
    x = copy.deepcopy(x_value)
    previousPoint = 0
    route_temp = [previousPoint]
    count = 0
    while(len(route_temp) < len(x)):
        if(x[previousPoint][count] > 0.99):
            previousPoint = count
            route_temp.append(previousPoint)
            count = 0
            continue
        else:
            count += 1
    return route_temp    




def solveTSP(truckcuss):
    # print("Function solveTSP() was called\n")
    global solveTSPcalledTime
    solveTSPcalledTime += 1 
    truckMkspn = 0
    truckRoute = []
    x0         = [0]
    backuppath = []
    for i in range(len(truckcuss)):
        x0.append(truckcuss[i].idNum)
        backuppath.append(truckcuss[i].idNum)
    x0.append(0) 
    
    # if(solveTSPcalledTime > 75):
    #     print("stop for check")
    
    
    
    
    N = len(x0)-1
    cost = [([0] * (N+1)) for p in range(N+1)]
    for i in range(N+1):
        for j in range(N+1):
            if(i != j):
                cost[i][j] = data.disMatrix[x0[i]][x0[j]]
                
    
    # if(solveTSPcalledTime > 75):
    #     print("stop for check")    
    
    
    # cost        = data.disMatrix
    if(len(truckcuss)<3):
        # print('return function was called, which means the num of truck route=2')
        x0 = x0[1:N]
        return [tsp_len(cost, x0), x0]
    model = Model('TSPbyGRB')
    model.setParam('OutputFlag', 0)
    X = {}
    mu = {}
    for i in range(N+1):
        mu[i] = model.addVar(lb = 0.0
                             , ub = 100
                             , vtype = GRB.CONTINUOUS
                             , name = "mu_" + str(i))
        for j in range(N+1):
            if(i != j):
                X[i, j] = model.addVar(vtype = GRB.BINARY
                                       , name = 'x_' + str(i)+'_'+str(j)
                                       )
        
    # set objective function
    obj = LinExpr(0)
    for key in X.keys():
        i = key[0]
        j = key[1]
        if(i < N and j < N):
            obj.addTerms(cost[key[0]][key[1]], X[key])
        elif(i == N):
            obj.addTerms(cost[0][key[1]], X[key])
        elif(j == N):
            obj.addTerms(cost[key[0]][0], X[key])
            
    model.setObjective(obj, GRB.MINIMIZE)
    
    # add constraints 1
    for j in range(1, N+1):
        lhs = LinExpr(0)
        for i in range(0, N):
            if(i!=j):
                lhs.addTerms(1, X[i, j])
        model.addConstr(lhs == 1, name = 'visit_' + str(j))
        
    # add constraint 2
    for i in range(0, N):
        lhs = LinExpr(0)
        for j in range(1, N + 1):
            if(i != j):
                lhs.addTerms(1, X[i, j])
        model.addConstr(lhs == 1, name = 'visit_' + str(j))
    
    for i in range(0, N):
        for j in range(1, N+1):
            if(i != j):
                model.addConstr(mu[i] - mu[j] + 100*X[i,j] <= 100-1)
                
    model.optimize()
    
    
    # if(solveTSPcalledTime > 75):
    #     print("stop for check")   
        
        
        
    x_value = getValue(X, N)
    truckRoute1 = getRoute(x_value)
    truckRoute1 = truckRoute1[1:N]
    
    for i in range(len(truckRoute1)):
        truckRoute.append(backuppath[truckRoute1[i]-1])
        
       
    truckMkspn = tsp_len(cost, truckRoute)
    return [truckMkspn, truckRoute]
            
    
def swap(umk, tmk, ua, tr): 
    # stand for: uavMkspn, truckMkspn, uavAssignments, truckRoute
    print("function SWAP was called, and uav Makespan, truck makespan, UAV customers and Truck customers' are: %.2f %.2f "%(umk,tmk) + str(ua) + str(tr))
       
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
    solveTSPcalledTime = 0
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
    print("solve TSP function called time: %d" %(solveTSPcalledTime))
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
    plt.title('Optimal Solution for PDSTSP heuristic (GRB4TSP)', font_dict)  
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
    
SA解TSP的PDSTSP启发式实现：

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
 
    
path = 'c101.txt'
customerNum1 = 100

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
需要指出的是，代码的实现由于原论文中只给出了数据集的生成原则，并没有数据集数据的具体信息。我们借用了solomn的数据集进行实验，刚开始主要在c101.txt上进行，txt文件的打开信息如下：

RANGE
20

LUNCTING RECOVER
1      1

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME
    0      40         50          0          0       1236          0   
    1      45         68         10        912        967         90   
    2      45         70         30        825        870         90   
    3      42         66         10         65        146         90   
    4      42         68         10        727        782         90   
    5      42         65         10         15         67         90   
    6      40         69         20        621        702         90   
    7      40         66         20        170        225         90   
    8      38         68         20        255        324         90   
    9      38         70         10        534        605         90   
   10      35         66         10        357        410         90   
   11      35         69         10        448        505         90   
   12      25         85         20        652        721         90   
   13      22         75         30         30         92         90   
   14      22         85         10        567        620         90   
   15      20         80         40        384        429         90   
   16      20         85         40        475        528         90   
   17      18         75         20         99        148         90   
   18      15         75         20        179        254         90   
   19      15         80         10        278        345         90   
   20      30         50         10         10         73         90   
   21      30         52         20        914        965         90   
   22      28         52         20        812        883         90   
   23      28         55         10        732        777         90   
   24      25         50         10         65        144         90   
   25      25         52         40        169        224         90   
   26      25         55         10        622        701         90   
   27      23         52         10        261        316         90   
   28      23         55         20        546        593         90   
   29      20         50         10        358        405         90   
   30      20         55         10        449        504         90   
   31      10         35         20        200        237         90   
   32      10         40         30         31        100         90   
   33       8         40         40         87        158         90   
   34       8         45         20        751        816         90   
   35       5         35         10        283        344         90   
   36       5         45         10        665        716         90   
   37       2         40         20        383        434         90   
   38       0         40         30        479        522         90   
   39       0         45         20        567        624         90   
   40      35         30         10        264        321         90   
   41      35         32         10        166        235         90   
   42      33         32         20         68        149         90   
   43      33         35         10         16         80         90   
   44      32         30         10        359        412         90   
   45      30         30         10        541        600         90   
   46      30         32         30        448        509         90   
   47      30         35         10       1054       1127         90   
   48      28         30         10        632        693         90   
   49      28         35         10       1001       1066         90   
   50      26         32         10        815        880         90   
   51      25         30         10        725        786         90   
   52      25         35         10        912        969         90   
   53      44          5         20        286        347         90   
   54      42         10         40        186        257         90   
   55      42         15         10         95        158         90   
   56      40          5         30        385        436         90   
   57      40         15         40         35         87         90   
   58      38          5         30        471        534         90   
   59      38         15         10        651        740         90   
   60      35          5         20        562        629         90   
   61      50         30         10        531        610         90   
   62      50         35         20        262        317         90   
   63      50         40         50        171        218         90   
   64      48         30         10        632        693         90   
   65      48         40         10         76        129         90   
   66      47         35         10        826        875         90   
   67      47         40         10         12         77         90   
   68      45         30         10        734        777         90   
   69      45         35         10        916        969         90   
   70      95         30         30        387        456         90   
   71      95         35         20        293        360         90   
   72      53         30         10        450        505         90   
   73      92         30         10        478        551         90   
   74      53         35         50        353        412         90   
   75      45         65         20        997       1068         90   
   76      90         35         10        203        260         90   
   77      88         30         10        574        643         90   
   78      88         35         20        109        170         90   
   79      87         30         10        668        731         90   
   80      85         25         10        769        820         90   
   81      85         35         30         47        124         90   
   82      75         55         20        369        420         90   
   83      72         55         10        265        338         90   
   84      70         58         20        458        523         90   
   85      68         60         30        555        612         90   
   86      66         55         10        173        238         90   
   87      65         55         20         85        144         90   
   88      65         60         30        645        708         90   
   89      63         58         10        737        802         90   
   90      60         55         10         20         84         90   
   91      60         60         10        836        889         90   
   92      67         85         20        368        441         90   
   93      65         85         40        475        518         90   
   94      65         82         10        285        336         90   
   95      62         80         30        196        239         90   
   96      60         80         10         95        156         90   
   97      60         85         30        561        622         90   
   98      58         75         20         30         84         90   
   99      55         80         10        743        820         90   
  100      55         85         20        647        726         90   
将结果可视化之后有：



 在算法对比中，SA启发式方法是在结果重复实验10次的场景下的最优解，可以看到在大多数情况下SA求解TSP的启发式都求得了问题的最优解，并且其求解的数据集顾客数为100时也是可以在652秒左右的时间求解完毕。

所有代码见：
flying2322/UAV-algorithm-design: PDSTSP problem algorithm design at Weihui (github.com)
https://github.com/flying2322/UAV-algorithm-design

solomn数据集：

solomn标准数据集，用于研究VRP问题_solomn数据集-交通文档类资源-CSDN文库
https://download.csdn.net/download/weixin_43464653/54077805

参考文献
【1】Murray, C. C., & Chu, A.he G. (2015). The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery. Transportation Research Part C: Emerging Technologies, 54, 86-109. https://doi.org/10.1016/j.trc.2015.03.005

【2】Min, L., Cheng, W., 1999. A genetic algorithm for minimizing the makespan in the case of scheduling identical parallel machines. Artif. Intell. Eng. 13 (4), 399–403.

【3】Xu, J., Nagi, R., 2013. Identical parallel machine scheduling to minimise makespan and total weighted completion time: a column generation approach. Int. J. Prod. Res. 51 (23–24), 7091–7104.

​
