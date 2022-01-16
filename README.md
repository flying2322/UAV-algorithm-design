# UAV-algorithm-design
**PDSTSP problem algorithm design at Weihui home**
## 1-11
new point about python programming
 1. copy.deepcopy([])
 2. random.sample(range(10), 3) 
 3. print(str(list=[]))




## 1-16 3:34
some results about 10 customers with only algorithm 6
### Output Window info for 10 Customers:
runfile('C:/CodeLi/pythonLi/PDSTSP/PDSTSP_heuristic.py', wdir='C:/CodeLi/pythonLi/PDSTSP')
 ***********Data Info***********

 UAV  range            =   20
 UAV's eligible CusNum =    8
 
 Test Markdown
|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |

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
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65  0.00 
18.68  0.00  2.00  3.61  3.00  4.24  5.10  5.39  7.00  7.28 10.20 10.05 18.68 
20.62  2.00  0.00  5.00  3.61  5.83  5.10  6.40  7.28  7.00 10.77 10.05 20.62 
16.12  3.61  5.00  0.00  2.00  1.00  3.61  2.00  4.47  5.66  7.00  7.62 16.12 
18.11  3.00  3.61  2.00  0.00  3.00  2.24  2.83  4.00  4.47  7.28  7.07 18.11 
15.13  4.24  5.83  1.00  3.00  0.00  4.47  2.24  5.00  6.40  7.07  8.06 15.13 
19.00  5.10  5.10  3.61  2.24  4.47  0.00  3.00  2.24  2.24  5.83  5.00 19.00 
16.00  5.39  6.40  2.00  2.83  2.24  3.00  0.00  2.83  4.47  5.00  5.83 16.00 
18.11  7.00  7.28  4.47  4.00  5.00  2.24  2.83  0.00  2.00  3.61  3.16 18.11 
20.10  7.28  7.00  5.66  4.47  6.40  2.24  4.47  2.00  0.00  5.00  3.16 20.10 
16.76 10.20 10.77  7.00  7.28  7.07  5.83  5.00  3.61  5.00  0.00  3.00 16.76 
19.65 10.05 10.05  7.62  7.07  8.06  5.00  5.83  3.16  3.16  3.00  0.00 19.65 
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65  0.00 

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
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65  0.00 
18.68  0.00  2.00  3.61  3.00  4.24  5.10  5.39  7.00  7.28 10.20 10.05 18.68 
20.62  2.00  0.00  5.00  3.61  5.83  5.10  6.40  7.28  7.00 10.77 10.05 20.62 
16.12  3.61  5.00  0.00  2.00  1.00  3.61  2.00  4.47  5.66  7.00  7.62 16.12 
18.11  3.00  3.61  2.00  0.00  3.00  2.24  2.83  4.00  4.47  7.28  7.07 18.11 
15.13  4.24  5.83  1.00  3.00  0.00  4.47  2.24  5.00  6.40  7.07  8.06 15.13 
19.00  5.10  5.10  3.61  2.24  4.47  0.00  3.00  2.24  2.24  5.83  5.00 19.00 
16.00  5.39  6.40  2.00  2.83  2.24  3.00  0.00  2.83  4.47  5.00  5.83 16.00 
18.11  7.00  7.28  4.47  4.00  5.00  2.24  2.83  0.00  2.00  3.61  3.16 18.11 
20.10  7.28  7.00  5.66  4.47  6.40  2.24  4.47  2.00  0.00  5.00  3.16 20.10 
16.76 10.20 10.77  7.00  7.28  7.07  5.83  5.00  3.61  5.00  0.00  3.00 16.76 
19.65 10.05 10.05  7.62  7.07  8.06  5.00  5.83  3.16  3.16  3.00  0.00 19.65 
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65  0.00 

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
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65 38.08 30.81 39.36 36.06  0.00 
18.68  0.00  2.00  3.61  3.00  4.24  5.10  5.39  7.00  7.28 10.20 10.05 26.25 24.04 28.60 27.73 18.68 
20.62  2.00  0.00  5.00  3.61  5.83  5.10  6.40  7.28  7.00 10.77 10.05 25.00 23.54 27.46 26.93 20.62 
16.12  3.61  5.00  0.00  2.00  1.00  3.61  2.00  4.47  5.66  7.00  7.62 25.50 21.93 27.59 26.08 16.12 
18.11  3.00  3.61  2.00  0.00  3.00  2.24  2.83  4.00  4.47  7.28  7.07 24.04 21.19 26.25 25.06 18.11 
15.13  4.24  5.83  1.00  3.00  0.00  4.47  2.24  5.00  6.40  7.07  8.06 26.25 22.36 28.28 26.63 15.13 
19.00  5.10  5.10  3.61  2.24  4.47  0.00  3.00  2.24  2.24  5.83  5.00 21.93 18.97 24.08 22.83 19.00 
16.00  5.39  6.40  2.00  2.83  2.24  3.00  0.00  2.83  4.47  5.00  5.83 24.21 20.12 26.17 24.41 16.00 
18.11  7.00  7.28  4.47  4.00  5.00  2.24  2.83  0.00  2.00  3.61  3.16 21.40 17.46 23.35 21.63 18.11 
20.10  7.28  7.00  5.66  4.47  6.40  2.24  4.47  2.00  0.00  5.00  3.16 19.85 16.76 21.93 20.59 20.10 
16.76 10.20 10.77  7.00  7.28  7.07  5.83  5.00  3.61  5.00  0.00  3.00 21.47 15.81 23.02 20.52 16.76 
19.65 10.05 10.05  7.62  7.07  8.06  5.00  5.83  3.16  3.16  3.00  0.00 18.87 14.32 20.62 18.60 19.65 
38.08 26.25 25.00 25.50 24.04 26.25 21.93 24.21 21.40 19.85 21.47 18.87  0.00 10.44  3.00  7.07 38.08 
30.81 24.04 23.54 21.93 21.19 22.36 18.97 20.12 17.46 16.76 15.81 14.32 10.44  0.00 10.00  5.39 30.81 
39.36 28.60 27.46 27.59 26.25 28.28 24.08 26.17 23.35 21.93 23.02 20.62  3.00 10.00  0.00  5.39 39.36 
36.06 27.73 26.93 26.08 25.06 26.63 22.83 24.41 21.63 20.59 20.52 18.60  7.07  5.39  5.39  0.00 36.06 
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65 38.08 30.81 39.36 36.06  0.00 

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
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65 38.08 30.81 39.36 36.06  0.00 
18.68  0.00  2.00  3.61  3.00  4.24  5.10  5.39  7.00  7.28 10.20 10.05 26.25 24.04 28.60 27.73 18.68 
20.62  2.00  0.00  5.00  3.61  5.83  5.10  6.40  7.28  7.00 10.77 10.05 25.00 23.54 27.46 26.93 20.62 
16.12  3.61  5.00  0.00  2.00  1.00  3.61  2.00  4.47  5.66  7.00  7.62 25.50 21.93 27.59 26.08 16.12 
18.11  3.00  3.61  2.00  0.00  3.00  2.24  2.83  4.00  4.47  7.28  7.07 24.04 21.19 26.25 25.06 18.11 
15.13  4.24  5.83  1.00  3.00  0.00  4.47  2.24  5.00  6.40  7.07  8.06 26.25 22.36 28.28 26.63 15.13 
19.00  5.10  5.10  3.61  2.24  4.47  0.00  3.00  2.24  2.24  5.83  5.00 21.93 18.97 24.08 22.83 19.00 
16.00  5.39  6.40  2.00  2.83  2.24  3.00  0.00  2.83  4.47  5.00  5.83 24.21 20.12 26.17 24.41 16.00 
18.11  7.00  7.28  4.47  4.00  5.00  2.24  2.83  0.00  2.00  3.61  3.16 21.40 17.46 23.35 21.63 18.11 
20.10  7.28  7.00  5.66  4.47  6.40  2.24  4.47  2.00  0.00  5.00  3.16 19.85 16.76 21.93 20.59 20.10 
16.76 10.20 10.77  7.00  7.28  7.07  5.83  5.00  3.61  5.00  0.00  3.00 21.47 15.81 23.02 20.52 16.76 
19.65 10.05 10.05  7.62  7.07  8.06  5.00  5.83  3.16  3.16  3.00  0.00 18.87 14.32 20.62 18.60 19.65 
38.08 26.25 25.00 25.50 24.04 26.25 21.93 24.21 21.40 19.85 21.47 18.87  0.00 10.44  3.00  7.07 38.08 
30.81 24.04 23.54 21.93 21.19 22.36 18.97 20.12 17.46 16.76 15.81 14.32 10.44  0.00 10.00  5.39 30.81 
39.36 28.60 27.46 27.59 26.25 28.28 24.08 26.17 23.35 21.93 23.02 20.62  3.00 10.00  0.00  5.39 39.36 
36.06 27.73 26.93 26.08 25.06 26.63 22.83 24.41 21.63 20.59 20.52 18.60  7.07  5.39  5.39  0.00 36.06 
 0.00 18.68 20.62 16.12 18.11 15.13 19.00 16.00 18.11 20.10 16.76 19.65 38.08 30.81 39.36 36.06  0.00 

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
