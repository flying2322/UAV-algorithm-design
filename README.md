


## 1-16 3:34
some results about 10 customers with only algorithm 6
### Output Window info for 10 Customers:
runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')
 ***********Data Info***********

 UAV  range            =   20
 UAV's eligible CusNum =    8
 
 Test Markdown  

*****************************Distance Matrix***************************
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19   | 16   | 18.11 | 20.1 | 16.76 | 0     |
|-------|-------|-------|-------|-------|-------|------|------|-------|------|-------|-------|
| 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1  | 5.39 | 7     | 7.28 | 10.2  | 18.68 |
| 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1  | 6.4  | 7.28  | 7    | 10.77 | 20.62 |
| 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61 | 2    | 4.47  | 5.66 | 7     | 16.12 |
| 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24 | 2.83 | 4     | 4.47 | 7.28  | 18.11 |
| 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47 | 2.24 | 5     | 6.4  | 7.07  | 15.13 |
| 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0    | 3    | 2.24  | 2.24 | 5.83  | 19    |
| 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3    | 0    | 2.83  | 4.47 | 5     | 16    |
| 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24 | 2.83 | 0     | 2    | 3.61  | 18.11 |
| 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24 | 4.47 | 2     | 0    | 5     | 20.1  |
| 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83 | 5    | 3.61  | 5    | 0     | 16.76 |
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19   | 16   | 18.11 | 20.1 | 16.76 | 0     |


maxSaving is :  17.03  
maxSaving is :  17.63  
maxSaving is :  18.10  
maxSaving is :  17.48  
maxSaving is :  14.15  
while loop break out, END!  
UAV makespan:  49.81  
Truck makespan:  51.43  
UAV Assignments:  
[1, 5, 7]  
Truck Assignments:  
[3, 4, 2, 6, 9, 8, 10]  
*********** ->PDSTSP heuristic algorithm<- ***********

![image](https://user-images.githubusercontent.com/48487718/149635780-132a52e8-9f6b-48cf-9143-f19ed57eaa3d.png)






---

some results about 11 customers with only algorithm 6
### Output Window info for 11 Customers:

runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')
 ***********Data Info***********

 UAV  range            =   20
 UAV's eligible CusNum =    9

*****************************Distance Matrix***************************
|       |       |       |       |       |       |      |      |       |      |       |       |       |
|-------|-------|-------|-------|-------|-------|------|------|-------|------|-------|-------|-------|
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19   | 16   | 18.11 | 20.1 | 16.76 | 19.65 | 0     |
| 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1  | 5.39 | 7     | 7.28 | 10.2  | 10.05 | 18.68 |
| 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1  | 6.4  | 7.28  | 7    | 10.77 | 10.05 | 20.62 |
| 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61 | 2    | 4.47  | 5.66 | 7     | 7.62  | 16.12 |
| 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24 | 2.83 | 4     | 4.47 | 7.28  | 7.07  | 18.11 |
| 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47 | 2.24 | 5     | 6.4  | 7.07  | 8.06  | 15.13 |
| 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0    | 3    | 2.24  | 2.24 | 5.83  | 5     | 19    |
| 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3    | 0    | 2.83  | 4.47 | 5     | 5.83  | 16    |
| 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24 | 2.83 | 0     | 2    | 3.61  | 3.16  | 18.11 |
| 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24 | 4.47 | 2     | 0    | 5     | 3.16  | 20.1  |
| 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83 | 5    | 3.61  | 5    | 0     | 3     | 16.76 |
| 19.65 | 10.05 | 10.05 | 7.62  | 7.07  | 8.06  | 5    | 5.83 | 3.16  | 3.16 | 3     | 0     | 19.65 |
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19   | 16   | 18.11 | 20.1 | 16.76 | 19.65 | 0     |


maxSaving is :  17.03  
maxSaving is :  17.63  
maxSaving is :  18.10  
maxSaving is :  17.48  
maxSaving is :  17.09  
maxSaving is :  14.15  
PDSTSP heuristic (Algorithm6) was successfully called!  

*************** The optimal solution are AS FOLLOWS: *************  

UAV makespan  :  49.81  
Truck makespan:  53.99  
UAV Assignments  :[1, 5, 7]  
Truck Assignments:[3, 4, 2, 6, 8, 9, 11, 10]  

******* Detailed path info was shown in PLOTS windows above! *****  

![image](https://user-images.githubusercontent.com/48487718/149635736-7b8d2b3a-530a-4e6d-a950-06633557e5fa.png)




runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')

 ***********Data Info***********

 UAV  range            =   20
 UAV's eligible CusNum =    9

*****************************Distance Matrix***************************
|       |       |       |       |       |       |      |      |       |      |       |       |       |
|-------|-------|-------|-------|-------|-------|------|------|-------|------|-------|-------|-------|
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19   | 16   | 18.11 | 20.1 | 16.76 | 19.65 | 0     |
| 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1  | 5.39 | 7     | 7.28 | 10.2  | 10.05 | 18.68 |
| 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1  | 6.4  | 7.28  | 7    | 10.77 | 10.05 | 20.62 |
| 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61 | 2    | 4.47  | 5.66 | 7     | 7.62  | 16.12 |
| 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24 | 2.83 | 4     | 4.47 | 7.28  | 7.07  | 18.11 |
| 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47 | 2.24 | 5     | 6.4  | 7.07  | 8.06  | 15.13 |
| 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0    | 3    | 2.24  | 2.24 | 5.83  | 5     | 19    |
| 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3    | 0    | 2.83  | 4.47 | 5     | 5.83  | 16    |
| 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24 | 2.83 | 0     | 2    | 3.61  | 3.16  | 18.11 |
| 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24 | 4.47 | 2     | 0    | 5     | 3.16  | 20.1  |
| 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83 | 5    | 3.61  | 5    | 0     | 3     | 16.76 |
| 19.65 | 10.05 | 10.05 | 7.62  | 7.07  | 8.06  | 5    | 5.83 | 3.16  | 3.16 | 3     | 0     | 19.65 |
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19   | 16   | 18.11 | 20.1 | 16.76 | 19.65 | 0     |




maxSaving is :  17.03  
maxSaving is :  17.63  
maxSaving is :  18.10  
maxSaving is :  17.48  
maxSaving is :  17.09  
maxSaving is :  15.12  
maxSaving is :  14.15  
PDSTSP heuristic (Algorithm6) was successfully called!  

*************** The optimal solution are AS FOLLOWS: *************  

UAV makespan  : 34.68  
Truck makespan: 54.00  
UAV Assignments  : [1, 7]  
Truck Assignments: [10, 11, 8, 9, 6, 2, 4, 3, 5]  

******* Detailed path info was shown in PLOTS windows above! *****  
![image](https://user-images.githubusercontent.com/48487718/149635551-57720c7b-f8ab-42f2-b43c-97e4cff0bddd.png)

---

some results about 15 customers with only algorithm 6  
### Output Window info for 15 Customers:  







runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')
 ***********Data Info***********  

 UAV  range            =   20  
 UAV's eligible CusNum =   10  

*****************************Distance Matrix***************************  
|       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 0     |
| 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1   | 5.39  | 7     | 7.28  | 10.2  | 10.05 | 26.25 | 24.04 | 28.6  | 27.73 | 18.68 |
| 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1   | 6.4   | 7.28  | 7     | 10.77 | 10.05 | 25    | 23.54 | 27.46 | 26.93 | 20.62 |
| 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61  | 2     | 4.47  | 5.66  | 7     | 7.62  | 25.5  | 21.93 | 27.59 | 26.08 | 16.12 |
| 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24  | 2.83  | 4     | 4.47  | 7.28  | 7.07  | 24.04 | 21.19 | 26.25 | 25.06 | 18.11 |
| 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47  | 2.24  | 5     | 6.4   | 7.07  | 8.06  | 26.25 | 22.36 | 28.28 | 26.63 | 15.13 |
| 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0     | 3     | 2.24  | 2.24  | 5.83  | 5     | 21.93 | 18.97 | 24.08 | 22.83 | 19    |
| 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3     | 0     | 2.83  | 4.47  | 5     | 5.83  | 24.21 | 20.12 | 26.17 | 24.41 | 16    |
| 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24  | 2.83  | 0     | 2     | 3.61  | 3.16  | 21.4  | 17.46 | 23.35 | 21.63 | 18.11 |
| 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24  | 4.47  | 2     | 0     | 5     | 3.16  | 19.85 | 16.76 | 21.93 | 20.59 | 20.1  |
| 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83  | 5     | 3.61  | 5     | 0     | 3     | 21.47 | 15.81 | 23.02 | 20.52 | 16.76 |
| 19.65 | 10.05 | 10.05 | 7.62  | 7.07  | 8.06  | 5     | 5.83  | 3.16  | 3.16  | 3     | 0     | 18.87 | 14.32 | 20.62 | 18.6  | 19.65 |
| 38.08 | 26.25 | 25    | 25.5  | 24.04 | 26.25 | 21.93 | 24.21 | 21.4  | 19.85 | 21.47 | 18.87 | 0     | 10.44 | 3     | 7.07  | 38.08 |
| 30.81 | 24.04 | 23.54 | 21.93 | 21.19 | 22.36 | 18.97 | 20.12 | 17.46 | 16.76 | 15.81 | 14.32 | 10.44 | 0     | 10    | 5.39  | 30.81 |
| 39.36 | 28.6  | 27.46 | 27.59 | 26.25 | 28.28 | 24.08 | 26.17 | 23.35 | 21.93 | 23.02 | 20.62 | 3     | 10    | 0     | 5.39  | 39.36 |
| 36.06 | 27.73 | 26.93 | 26.08 | 25.06 | 26.63 | 22.83 | 24.41 | 21.63 | 20.59 | 20.52 | 18.6  | 7.07  | 5.39  | 5.39  | 0     | 36.06 |
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 0     |


maxSaving is :  37.84  
maxSaving is :  18.96  
maxSaving is :  19.95  
maxSaving is :  17.48  
maxSaving is :  15.12  
maxSaving is :  16.11  
while loop break out, END!  
UAV makespan:  74.94  
Truck makespan:  90.80  
UAV Assignments:  
[1, 2, 7, 11]  
Truck Assignments:  
[10, 13, 15, 14, 12, 9, 8, 6, 4, 3, 5]  
*********** ->PDSTSP heuristic algorithm<- ***********  
![image](https://user-images.githubusercontent.com/48487718/149635798-266dfdfc-f278-4621-ba1c-ba57d5abdd18.png)  



Removing all variables...   
 

runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')  
 ***********Data Info***********  

 UAV  range            =   20  
 UAV's eligible CusNum =   10  

*****************************Distance Matrix***************************  
|       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 0     |
| 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1   | 5.39  | 7     | 7.28  | 10.2  | 10.05 | 26.25 | 24.04 | 28.6  | 27.73 | 18.68 |
| 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1   | 6.4   | 7.28  | 7     | 10.77 | 10.05 | 25    | 23.54 | 27.46 | 26.93 | 20.62 |
| 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61  | 2     | 4.47  | 5.66  | 7     | 7.62  | 25.5  | 21.93 | 27.59 | 26.08 | 16.12 |
| 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24  | 2.83  | 4     | 4.47  | 7.28  | 7.07  | 24.04 | 21.19 | 26.25 | 25.06 | 18.11 |
| 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47  | 2.24  | 5     | 6.4   | 7.07  | 8.06  | 26.25 | 22.36 | 28.28 | 26.63 | 15.13 |
| 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0     | 3     | 2.24  | 2.24  | 5.83  | 5     | 21.93 | 18.97 | 24.08 | 22.83 | 19    |
| 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3     | 0     | 2.83  | 4.47  | 5     | 5.83  | 24.21 | 20.12 | 26.17 | 24.41 | 16    |
| 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24  | 2.83  | 0     | 2     | 3.61  | 3.16  | 21.4  | 17.46 | 23.35 | 21.63 | 18.11 |
| 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24  | 4.47  | 2     | 0     | 5     | 3.16  | 19.85 | 16.76 | 21.93 | 20.59 | 20.1  |
| 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83  | 5     | 3.61  | 5     | 0     | 3     | 21.47 | 15.81 | 23.02 | 20.52 | 16.76 |
| 19.65 | 10.05 | 10.05 | 7.62  | 7.07  | 8.06  | 5     | 5.83  | 3.16  | 3.16  | 3     | 0     | 18.87 | 14.32 | 20.62 | 18.6  | 19.65 |
| 38.08 | 26.25 | 25    | 25.5  | 24.04 | 26.25 | 21.93 | 24.21 | 21.4  | 19.85 | 21.47 | 18.87 | 0     | 10.44 | 3     | 7.07  | 38.08 |
| 30.81 | 24.04 | 23.54 | 21.93 | 21.19 | 22.36 | 18.97 | 20.12 | 17.46 | 16.76 | 15.81 | 14.32 | 10.44 | 0     | 10    | 5.39  | 30.81 |
| 39.36 | 28.6  | 27.46 | 27.59 | 26.25 | 28.28 | 24.08 | 26.17 | 23.35 | 21.93 | 23.02 | 20.62 | 3     | 10    | 0     | 5.39  | 39.36 |
| 36.06 | 27.73 | 26.93 | 26.08 | 25.06 | 26.63 | 22.83 | 24.41 | 21.63 | 20.59 | 20.52 | 18.6  | 7.07  | 5.39  | 5.39  | 0     | 36.06 |
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 0     |


maxSaving is :  35.96  
maxSaving is :  21.30  
maxSaving is :  18.82  
maxSaving is :  17.48  
maxSaving is :  15.12  
maxSaving is :  17.47  
while loop break out, END!  
UAV makespan:  73.41  
Truck makespan:  90.98  
UAV Assignments:  
[1, 2, 7, 8]  
Truck Assignments:  
[10, 13, 15, 14, 12, 11, 9, 6, 4, 3, 5]  
*********** ->PDSTSP heuristic algorithm<- ***********  
### Output Window info for 20 Customers:    

runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')  
 ***********Data Info***********  

 UAV  range            =   20  
 Customer's     number =   20  
 UAV's eligible CusNum =   10    

*****************************Distance Matrix***************************  

| 0  | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    | 21    |
|----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0  | 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 40.31 | 33.3  | 35.36 | 39.05 | 10    | 0     |
| 1  | 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1   | 5.39  | 7     | 7.28  | 10.2  | 10.05 | 26.25 | 24.04 | 28.6  | 27.73 | 30.23 | 27.89 | 30.81 | 32.31 | 23.43 | 18.68 |
| 2  | 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1   | 6.4   | 7.28  | 7     | 10.77 | 10.05 | 25    | 23.54 | 27.46 | 26.93 | 29.15 | 27.46 | 30.41 | 31.62 | 25    | 20.62 |
| 3  | 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61  | 2     | 4.47  | 5.66  | 7     | 7.62  | 25.5  | 21.93 | 27.59 | 26.08 | 29.07 | 25.63 | 28.46 | 30.41 | 20    | 16.12 |
| 4  | 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24  | 2.83  | 4     | 4.47  | 7.28  | 7.07  | 24.04 | 21.19 | 26.25 | 25.06 | 27.8  | 25    | 27.89 | 29.55 | 21.63 | 18.11 |
| 5  | 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47  | 2.24  | 5     | 6.4   | 7.07  | 8.06  | 26.25 | 22.36 | 28.28 | 26.63 | 29.73 | 26    | 28.79 | 30.89 | 19.21 | 15.13 |
| 6  | 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0     | 3     | 2.24  | 2.24  | 5.83  | 5     | 21.93 | 18.97 | 24.08 | 22.83 | 25.61 | 22.8  | 25.71 | 27.31 | 21.47 | 19    |
| 7  | 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3     | 0     | 2.83  | 4.47  | 5     | 5.83  | 24.21 | 20.12 | 26.17 | 24.41 | 27.59 | 23.77 | 26.57 | 28.65 | 18.87 | 16    |
| 8  | 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24  | 2.83  | 0     | 2     | 3.61  | 3.16  | 21.4  | 17.46 | 23.35 | 21.63 | 24.76 | 21.19 | 24.04 | 25.94 | 19.7  | 18.11 |
| 9  | 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24  | 4.47  | 2     | 0     | 5     | 3.16  | 19.85 | 16.76 | 21.93 | 20.59 | 23.43 | 20.62 | 23.54 | 25.08 | 21.54 | 20.1  |
| 10 | 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83  | 5     | 3.61  | 5     | 0     | 3     | 21.47 | 15.81 | 23.02 | 20.52 | 24.21 | 19.24 | 21.93 | 24.41 | 16.76 | 16.76 |
| 11 | 19.65 | 10.05 | 10.05 | 7.62  | 7.07  | 8.06  | 5     | 5.83  | 3.16  | 3.16  | 3     | 0     | 18.87 | 14.32 | 20.62 | 18.6  | 21.93 | 18.03 | 20.88 | 22.83 | 19.65 | 19.65 |
| 12 | 38.08 | 26.25 | 25    | 25.5  | 24.04 | 26.25 | 21.93 | 24.21 | 21.4  | 19.85 | 21.47 | 18.87 | 0     | 10.44 | 3     | 7.07  | 5     | 12.21 | 14.14 | 11.18 | 35.36 | 38.08 |
| 13 | 30.81 | 24.04 | 23.54 | 21.93 | 21.19 | 22.36 | 18.97 | 20.12 | 17.46 | 16.76 | 15.81 | 14.32 | 10.44 | 0     | 10    | 5.39  | 10.2  | 4     | 7     | 8.6   | 26.25 | 30.81 |
| 14 | 39.36 | 28.6  | 27.46 | 27.59 | 26.25 | 28.28 | 24.08 | 26.17 | 23.35 | 21.93 | 23.02 | 20.62 | 3     | 10    | 0     | 5.39  | 2     | 10.77 | 12.21 | 8.6   | 35.9  | 39.36 |
| 15 | 36.06 | 27.73 | 26.93 | 26.08 | 25.06 | 26.63 | 22.83 | 24.41 | 21.63 | 20.59 | 20.52 | 18.6  | 7.07  | 5.39  | 5.39  | 0     | 5     | 5.39  | 7.07  | 5     | 31.62 | 36.06 |
| 16 | 40.31 | 30.23 | 29.15 | 29.07 | 27.8  | 29.73 | 25.61 | 27.59 | 24.76 | 23.43 | 24.21 | 21.93 | 5     | 10.2  | 2     | 5     | 0     | 10.2  | 11.18 | 7.07  | 36.4  | 40.31 |
| 17 | 33.3  | 27.89 | 27.46 | 25.63 | 25    | 26    | 22.8  | 23.77 | 21.19 | 20.62 | 19.24 | 18.03 | 12.21 | 4     | 10.77 | 5.39  | 10.2  | 0     | 3     | 5.83  | 27.73 | 33.3  |
| 18 | 35.36 | 30.81 | 30.41 | 28.46 | 27.89 | 28.79 | 25.71 | 26.57 | 24.04 | 23.54 | 21.93 | 20.88 | 14.14 | 7     | 12.21 | 7.07  | 11.18 | 3     | 0     | 5     | 29.15 | 35.36 |
| 19 | 39.05 | 32.31 | 31.62 | 30.41 | 29.55 | 30.89 | 27.31 | 28.65 | 25.94 | 25.08 | 24.41 | 22.83 | 11.18 | 8.6   | 8.6   | 5     | 7.07  | 5.83  | 5     | 0     | 33.54 | 39.05 |
| 20 | 10    | 23.43 | 25    | 20    | 21.63 | 19.21 | 21.47 | 18.87 | 19.7  | 21.54 | 16.76 | 19.65 | 35.36 | 26.25 | 35.9  | 31.62 | 36.4  | 27.73 | 29.15 | 33.54 | 0     | 10    |
| 21 | 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 40.31 | 33.3  | 35.36 | 39.05 | 10    | 0     |

maxSaving is :  31.45  
maxSaving is :  20.08  
maxSaving is :  17.03  
maxSaving is :  17.63  
function SWAP was called, and uav Makespan, truck makespan, UAV customers and Truck customers' are: 108.64 113.15   
[1, 2, 4, 5, 7, 9][20, 17, 18, 19, 16, 14, 12, 15, 13, 11, 10, 8, 6, 3]  
function SWAP was called, and uav Makespan, truck makespan, UAV customers and Truck customers' are: 111.64 112.13   
[1, 2, 4, 5, 9, 6][20, 17, 18, 19, 16, 14, 12, 15, 13, 11, 10, 8, 7, 3]  
PDSTSP heuristic (Algorithm6) was successfully called!  
   
*************** The optimal solution are AS FOLLOWS: *************  
  
UAV makespan  : 111.64  
Truck makespan: 112.13  
UAV Assignments  : [1, 2, 4, 5, 9, 6]  
Truck Assignments: [20, 17, 18, 19, 16, 14, 12, 15, 13, 11, 10, 8, 7, 3]  

******* Detailed path info was shown in PLOTS windows above! *****

![image](https://user-images.githubusercontent.com/48487718/149655632-67817855-a7ea-44a9-9659-f79b54510dde.png)  

### Output Window info for 30 Customers:  
runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')  
 ***********Data Info***********  

 UAV  range            =   20  
 Customer's     number =   30  
 UAV's eligible CusNum =   20  

*****************************Distance Matrix***************************  
|       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 40.31 | 33.3  | 35.36 | 39.05 | 10    | 10.2  | 12.17 | 13    | 15    | 15.13 | 15.81 | 17.12 | 17.72 | 20    | 20.62 | 0     |
| 18.68 | 0     | 2     | 3.61  | 3     | 4.24  | 5.1   | 5.39  | 7     | 7.28  | 10.2  | 10.05 | 26.25 | 24.04 | 28.6  | 27.73 | 30.23 | 27.89 | 30.81 | 32.31 | 23.43 | 21.93 | 23.35 | 21.4  | 26.91 | 25.61 | 23.85 | 27.2  | 25.55 | 30.81 | 28.18 | 18.68 |
| 20.62 | 2     | 0     | 5     | 3.61  | 5.83  | 5.1   | 6.4   | 7.28  | 7     | 10.77 | 10.05 | 25    | 23.54 | 27.46 | 26.93 | 29.15 | 27.46 | 30.41 | 31.62 | 25    | 23.43 | 24.76 | 22.67 | 28.28 | 26.91 | 25    | 28.43 | 26.63 | 32.02 | 29.15 | 20.62 |
| 16.12 | 3.61  | 5     | 0     | 2     | 1     | 3.61  | 2     | 4.47  | 5.66  | 7     | 7.62  | 25.5  | 21.93 | 27.59 | 26.08 | 29.07 | 25.63 | 28.46 | 30.41 | 20    | 18.44 | 19.8  | 17.8  | 23.35 | 22.02 | 20.25 | 23.6  | 21.95 | 27.2  | 24.6  | 16.12 |
| 18.11 | 3     | 3.61  | 2     | 0     | 3     | 2.24  | 2.83  | 4     | 4.47  | 7.28  | 7.07  | 24.04 | 21.19 | 26.25 | 25.06 | 27.8  | 25    | 27.89 | 29.55 | 21.63 | 20    | 21.26 | 19.1  | 24.76 | 23.35 | 21.4  | 24.84 | 23.02 | 28.43 | 25.55 | 18.11 |
| 15.13 | 4.24  | 5.83  | 1     | 3     | 0     | 4.47  | 2.24  | 5     | 6.4   | 7.07  | 8.06  | 26.25 | 22.36 | 28.28 | 26.63 | 29.73 | 26    | 28.79 | 30.89 | 19.21 | 17.69 | 19.1  | 17.2  | 22.67 | 21.4  | 19.72 | 23.02 | 21.47 | 26.63 | 24.17 | 15.13 |
| 19    | 5.1   | 5.1   | 3.61  | 2.24  | 4.47  | 0     | 3     | 2.24  | 2.24  | 5.83  | 5     | 21.93 | 18.97 | 24.08 | 22.83 | 25.61 | 22.8  | 25.71 | 27.31 | 21.47 | 19.72 | 20.81 | 18.44 | 24.21 | 22.67 | 20.52 | 24.04 | 22.02 | 27.59 | 24.41 | 19    |
| 16    | 5.39  | 6.4   | 2     | 2.83  | 2.24  | 3     | 0     | 2.83  | 4.47  | 5     | 5.83  | 24.21 | 20.12 | 26.17 | 24.41 | 27.59 | 23.77 | 26.57 | 28.65 | 18.87 | 17.2  | 18.44 | 16.28 | 21.93 | 20.52 | 18.6  | 22.02 | 20.25 | 25.61 | 22.83 | 16    |
| 18.11 | 7     | 7.28  | 4.47  | 4     | 5     | 2.24  | 2.83  | 0     | 2     | 3.61  | 3.16  | 21.4  | 17.46 | 23.35 | 21.63 | 24.76 | 21.19 | 24.04 | 25.94 | 19.7  | 17.89 | 18.87 | 16.4  | 22.2  | 20.62 | 18.38 | 21.93 | 19.85 | 25.46 | 22.2  | 18.11 |
| 20.1  | 7.28  | 7     | 5.66  | 4.47  | 6.4   | 2.24  | 4.47  | 2     | 0     | 5     | 3.16  | 19.85 | 16.76 | 21.93 | 20.59 | 23.43 | 20.62 | 23.54 | 25.08 | 21.54 | 19.7  | 20.59 | 18.03 | 23.85 | 22.2  | 19.85 | 23.43 | 21.21 | 26.91 | 23.43 | 20.1  |
| 16.76 | 10.2  | 10.77 | 7     | 7.28  | 7.07  | 5.83  | 5     | 3.61  | 5     | 0     | 3     | 21.47 | 15.81 | 23.02 | 20.52 | 24.21 | 19.24 | 21.93 | 24.41 | 16.76 | 14.87 | 15.65 | 13.04 | 18.87 | 17.2  | 14.87 | 18.44 | 16.28 | 21.93 | 18.6  | 16.76 |
| 19.65 | 10.05 | 10.05 | 7.62  | 7.07  | 8.06  | 5     | 5.83  | 3.16  | 3.16  | 3     | 0     | 18.87 | 14.32 | 20.62 | 18.6  | 21.93 | 18.03 | 20.88 | 22.83 | 19.65 | 17.72 | 18.38 | 15.65 | 21.47 | 19.72 | 17.2  | 20.81 | 18.44 | 24.21 | 20.52 | 19.65 |
| 38.08 | 26.25 | 25    | 25.5  | 24.04 | 26.25 | 21.93 | 24.21 | 21.4  | 19.85 | 21.47 | 18.87 | 0     | 10.44 | 3     | 7.07  | 5     | 12.21 | 14.14 | 11.18 | 35.36 | 33.38 | 33.14 | 30.15 | 35    | 33    | 30    | 33.06 | 30.07 | 35.36 | 30.41 | 38.08 |
| 30.81 | 24.04 | 23.54 | 21.93 | 21.19 | 22.36 | 18.97 | 20.12 | 17.46 | 16.76 | 15.81 | 14.32 | 10.44 | 0     | 10    | 5.39  | 10.2  | 4     | 7     | 8.6   | 26.25 | 24.35 | 23.77 | 20.88 | 25.18 | 23.19 | 20.22 | 23.02 | 20.02 | 25.08 | 20.1  | 30.81 |
| 39.36 | 28.6  | 27.46 | 27.59 | 26.25 | 28.28 | 24.08 | 26.17 | 23.35 | 21.93 | 23.02 | 20.62 | 3     | 10    | 0     | 5.39  | 2     | 10.77 | 12.21 | 8.6   | 35.9  | 33.96 | 33.54 | 30.59 | 35.13 | 33.14 | 30.15 | 33.02 | 30.02 | 35.06 | 30.07 | 39.36 |
| 36.06 | 27.73 | 26.93 | 26.08 | 25.06 | 26.63 | 22.83 | 24.41 | 21.63 | 20.59 | 20.52 | 18.6  | 7.07  | 5.39  | 5.39  | 0     | 5     | 5.39  | 7.07  | 5     | 31.62 | 29.73 | 29.12 | 26.25 | 30.41 | 28.44 | 25.5  | 28.16 | 25.18 | 30    | 25    | 36.06 |
| 40.31 | 30.23 | 29.15 | 29.07 | 27.8  | 29.73 | 25.61 | 27.59 | 24.76 | 23.43 | 24.21 | 21.93 | 5     | 10.2  | 2     | 5     | 0     | 10.2  | 11.18 | 7.07  | 36.4  | 34.48 | 33.96 | 31.05 | 35.36 | 33.38 | 30.41 | 33.14 | 30.15 | 35    | 30    | 40.31 |
| 33.3  | 27.89 | 27.46 | 25.63 | 25    | 26    | 22.8  | 23.77 | 21.19 | 20.62 | 19.24 | 18.03 | 12.21 | 4     | 10.77 | 5.39  | 10.2  | 0     | 3     | 5.83  | 27.73 | 25.94 | 25.08 | 22.36 | 25.96 | 24.04 | 21.19 | 23.54 | 20.62 | 25.08 | 20.1  | 33.3  |
| 35.36 | 30.81 | 30.41 | 28.46 | 27.89 | 28.79 | 25.71 | 26.57 | 24.04 | 23.54 | 21.93 | 20.88 | 14.14 | 7     | 12.21 | 7.07  | 11.18 | 3     | 0     | 5     | 29.15 | 27.46 | 26.42 | 23.85 | 26.93 | 25.08 | 22.36 | 24.35 | 21.54 | 25.5  | 20.62 | 35.36 |
| 39.05 | 32.31 | 31.62 | 30.41 | 29.55 | 30.89 | 27.31 | 28.65 | 25.94 | 25.08 | 24.41 | 22.83 | 11.18 | 8.6   | 8.6   | 5     | 7.07  | 5.83  | 5     | 0     | 33.54 | 31.76 | 30.87 | 28.18 | 31.62 | 29.73 | 26.93 | 29.12 | 26.25 | 30.41 | 25.5  | 39.05 |
| 10    | 23.43 | 25    | 20    | 21.63 | 19.21 | 21.47 | 18.87 | 19.7  | 21.54 | 16.76 | 19.65 | 35.36 | 26.25 | 35.9  | 31.62 | 36.4  | 27.73 | 29.15 | 33.54 | 0     | 2     | 2.83  | 5.39  | 5     | 5.39  | 7.07  | 7.28  | 8.6   | 10    | 11.18 | 10    |
| 10.2  | 21.93 | 23.43 | 18.44 | 20    | 17.69 | 19.72 | 17.2  | 17.89 | 19.7  | 14.87 | 17.72 | 33.38 | 24.35 | 33.96 | 29.73 | 34.48 | 25.94 | 27.46 | 31.76 | 2     | 0     | 2     | 3.61  | 5.39  | 5     | 5.83  | 7     | 7.62  | 10.2  | 10.44 | 10.2  |
| 12.17 | 23.35 | 24.76 | 19.8  | 21.26 | 19.1  | 20.81 | 18.44 | 18.87 | 20.59 | 15.65 | 18.38 | 33.14 | 23.77 | 33.54 | 29.12 | 33.96 | 25.08 | 26.42 | 30.87 | 2.83  | 2     | 0     | 3     | 3.61  | 3     | 4.24  | 5     | 5.83  | 8.25  | 8.54  | 12.17 |
| 13    | 21.4  | 22.67 | 17.8  | 19.1  | 17.2  | 18.44 | 16.28 | 16.4  | 18.03 | 13.04 | 15.65 | 30.15 | 20.88 | 30.59 | 26.25 | 31.05 | 22.36 | 23.85 | 28.18 | 5.39  | 3.61  | 3     | 0     | 5.83  | 4.24  | 3     | 5.83  | 5     | 9.43  | 8     | 13    |
| 15    | 26.91 | 28.28 | 23.35 | 24.76 | 22.67 | 24.21 | 21.93 | 22.2  | 23.85 | 18.87 | 21.47 | 35    | 25.18 | 35.13 | 30.41 | 35.36 | 25.96 | 26.93 | 31.62 | 5     | 5.39  | 3.61  | 5.83  | 0     | 2     | 5     | 2.83  | 5.39  | 5     | 7.07  | 15    |
| 15.13 | 25.61 | 26.91 | 22.02 | 23.35 | 21.4  | 22.67 | 20.52 | 20.62 | 22.2  | 17.2  | 19.72 | 33    | 23.19 | 33.14 | 28.44 | 33.38 | 24.04 | 25.08 | 29.73 | 5.39  | 5     | 3     | 4.24  | 2     | 0     | 3     | 2     | 3.61  | 5.39  | 5.83  | 15.13 |
| 15.81 | 23.85 | 25    | 20.25 | 21.4  | 19.72 | 20.52 | 18.6  | 18.38 | 19.85 | 14.87 | 17.2  | 30    | 20.22 | 30.15 | 25.5  | 30.41 | 21.19 | 22.36 | 26.93 | 7.07  | 5.83  | 4.24  | 3     | 5     | 3     | 0     | 3.61  | 2     | 7.07  | 5     | 15.81 |
| 17.12 | 27.2  | 28.43 | 23.6  | 24.84 | 23.02 | 24.04 | 22.02 | 21.93 | 23.43 | 18.44 | 20.81 | 33.06 | 23.02 | 33.02 | 28.16 | 33.14 | 23.54 | 24.35 | 29.12 | 7.28  | 7     | 5     | 5.83  | 2.83  | 2     | 3.61  | 0     | 3     | 3.61  | 4.24  | 17.12 |
| 17.72 | 25.55 | 26.63 | 21.95 | 23.02 | 21.47 | 22.02 | 20.25 | 19.85 | 21.21 | 16.28 | 18.44 | 30.07 | 20.02 | 30.02 | 25.18 | 30.15 | 20.62 | 21.54 | 26.25 | 8.6   | 7.62  | 5.83  | 5     | 5.39  | 3.61  | 2     | 3     | 0     | 5.83  | 3     | 17.72 |
| 20    | 30.81 | 32.02 | 27.2  | 28.43 | 26.63 | 27.59 | 25.61 | 25.46 | 26.91 | 21.93 | 24.21 | 35.36 | 25.08 | 35.06 | 30    | 35    | 25.08 | 25.5  | 30.41 | 10    | 10.2  | 8.25  | 9.43  | 5     | 5.39  | 7.07  | 3.61  | 5.83  | 0     | 5     | 20    |
| 20.62 | 28.18 | 29.15 | 24.6  | 25.55 | 24.17 | 24.41 | 22.83 | 22.2  | 23.43 | 18.6  | 20.52 | 30.41 | 20.1  | 30.07 | 25    | 30    | 20.1  | 20.62 | 25.5  | 11.18 | 10.44 | 8.54  | 8     | 7.07  | 5.83  | 5     | 4.24  | 3     | 5     | 0     | 20.62 |
| 0     | 18.68 | 20.62 | 16.12 | 18.11 | 15.13 | 19    | 16    | 18.11 | 20.1  | 16.76 | 19.65 | 38.08 | 30.81 | 39.36 | 36.06 | 40.31 | 33.3  | 35.36 | 39.05 | 10    | 10.2  | 12.17 | 13    | 15    | 15.13 | 15.81 | 17.12 | 17.72 | 20    | 20.62 | 0     |

maxSaving is :  27.55  
maxSaving is :  22.07  
maxSaving is :  18.14  
maxSaving is :  18.13  
maxSaving is :  17.03  
maxSaving is :  17.63  
maxSaving is :  18.10  
maxSaving is :  17.48  
maxSaving is :  15.12  
maxSaving is :  14.74  
maxSaving is :  14.15  
maxSaving is :  17.94  
function SWAP was called, and uav Makespan, truck makespan, UAV customers and Truck customers' are: 119.29 127.36   
[7, 21, 22, 23, 24, 26, 27, 29][20, 25, 28, 30, 17, 18, 19, 16, 14, 12, 15, 13, 11, 10, 8, 9, 6, 4, 2, 1, 3, 5]  
function SWAP was called, and uav Makespan, truck makespan, UAV customers and Truck customers' are: 119.89 126.99   
[7, 21, 22, 23, 24, 26, 29, 28][20, 25, 27, 30, 17, 18, 19, 16, 14, 12, 15, 13, 11, 10, 8, 9, 6, 4, 2, 1, 3, 5]  
PDSTSP heuristic (Algorithm6) was successfully called!  

*************** The optimal solution are AS FOLLOWS: *************  
  
UAV makespan  : 119.89  
Truck makespan: 126.99  
UAV Assignments  : [7, 21, 22, 23, 24, 26, 29, 28]  
Truck Assignments: [20, 25, 27, 30, 17, 18, 19, 16, 14, 12, 15, 13, 11, 10, 8, 9, 6, 4, 2, 1, 3, 5]  
![image](https://user-images.githubusercontent.com/48487718/149655622-40cecc2a-6c5e-4b8e-920b-f6854e92a579.png)

******* Detailed path info was shown in PLOTS windows above! *****  
