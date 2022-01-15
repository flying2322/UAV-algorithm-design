# UAV-algorithm-design
PDSTSP problem algorithm design at Weihui
## 1-11
new learning
 1. copy.deepcopy([])
 2. random.sample(range(10), 3) 

## 1-16 3:34
some results about 11 customers with only algorithm 6
### Output Window info:

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

