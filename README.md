​
本文复现的是论文【1】的第二部分PDSTSP问题的求解：The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery - 百度学术 (baidu.com)

 本文的代码复现参考了师兄的这篇帖子：

论文代码复现 | 无人机与卡车联合配送(Python+Gurobi)(The flying sidekick traveling salesman problem)_HsinglukLiu的博客-CSDN博客

因为在本论文中作者提出了两个模型，一是有无人机辅助的旅行商（The flying sidekick traveling salesman problem）问题（简记：FSTSP），二是并行无人机的调度优化问题（Parallel drone scheduling TSP, 简记为：PDSTSP）。这两类问题都相当于利用无人机的灵活配送效率高的优势进行建模，论文构建了两类问题的数学规划模型。

简单理解第一种建模FSTSP问题是将无人机和卡车在配送过程中进行协同，如下图【Fig.3】所示，无人机是从卡车在配送过程中的顾客点进行起降，同时在论文【1】中，为了简化问题，无人机路径只能和卡车路径组成三角形，即在卡车连续服务的两个顾客间无人机必须完成顾客服务的往返,例如图中的【4-7-6】和【6-1-5】。
![image](https://user-images.githubusercontent.com/48487718/153694188-b201424d-9f91-4d7b-aa48-9febd9b5aa70.png)



 第二种建模方式PDSTSP问题是指无人机和卡车彼此并行作业，当任务分配完成后，即不再有无人机和卡车的交互,见下图【Fig.2】。例如在图Fig.2（a）中的路径时间明显大于图b和图c，该问题相当于在初始给定一定量的顾客后我们进行两个决策，第一是哪些顾客交由无人机进行服务，哪些顾客交由卡车服务，第二部，在给定的顾客分配下，确定无人机和卡车的最优服务的顺序和路径。本文正是复现的PDSTSP问题的代码。
![image](https://user-images.githubusercontent.com/48487718/153694207-b88bba0c-0706-4753-8e41-a184b3043f1c.png)


# 详细内容参见链接：
https://blog.csdn.net/weixin_43464653/article/details/122877907

在算法对比中，SA启发式方法是在结果重复实验10次的场景下的最优解，可以看到在大多数情况下SA求解TSP的启发式都求得了问题的最优解，并且其求解的数据集顾客数为100时也是可以在652秒左右的时间求解完毕。

solomn数据集：
https://download.csdn.net/download/weixin_43464653/54077805

参考文献
【1】Murray, C. C., & Chu, A.he G. (2015). The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery. Transportation Research Part C: Emerging Technologies, 54, 86-109. https://doi.org/10.1016/j.trc.2015.03.005

【2】Min, L., Cheng, W., 1999. A genetic algorithm for minimizing the makespan in the case of scheduling identical parallel machines. Artif. Intell. Eng. 13 (4), 399–403.

【3】Xu, J., Nagi, R., 2013. Identical parallel machine scheduling to minimise makespan and total weighted completion time: a column generation approach. Int. J. Prod. Res. 51 (23–24), 7091–7104.

​
