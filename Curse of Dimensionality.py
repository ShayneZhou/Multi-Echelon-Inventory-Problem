# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:18:24 2020

@author: ShayneZhou


"""

import numpy as np
import matplotlib.pyplot as plt
 
#初始化全局参数
T = [2,3,4,5,6] #周期数
c = 10 #订货单价
h = 5 #库存持有成本
b = 40 #补货单价
alpha = 0.9 #折扣因子
Dmean = 2 #需求的均值
Dmax = 4 #需求的上限
Zmax = 6 #仓库的订单上限

#首先生成需求序列，0到10的序列，以及对应的概率值
demand = np.arange(Dmax + 1)

#验证维数灾难
y = []
for s in T:
    count = 0
    for t in range(s, 0, -1):
        for x3 in range(-Dmax * (t - 1), Zmax * (t - 1) + 1):#一直不订货且需求最大，一直订货且下游不订货无需求 
            for x2 in range(-Dmax * (t - 1), x3 + 1):#一直不订货且需求最大，总将第三级仓库的余量订空且无需求
                for x1 in range(-Dmax * (t - 1), x2 + 1):#一直不订货且需求最大，总将第二级仓库的余量订空且无需求
                    for z in range(0, Zmax + 1):#向外部订货量是从0到最大订货量
                        for q2 in range(0, x3 - x2 + z + 1):#向三级仓库订货量从0到其当前库存上限
                            for q1 in range(0, x2 - x1 + q2 + 1):#向二级仓库订货量从0到其当前库存上限
                                for d in demand:
                                    count = count + 1
    y.append(count)


#作图
plt.plot(T, y)
plt.title('Curse of Dimensionality')
plt.xlabel('Periods')
plt.ylabel('Cycle Times')
plt.show()