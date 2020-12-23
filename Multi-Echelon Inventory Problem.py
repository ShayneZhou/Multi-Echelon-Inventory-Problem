# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:18:24 2020

@author: ShayneZhou

三级库存，第一级面向需求，第三级面向供应
每一期成本包含ordering cost + holding cost + penalty cost
假设提前期为0
backlog方法验证
假设需求是泊松分布
V记录了每一期的不同状态的成本
optimal_action记录了不同时期不同状态下的最优决策
Policy Structure1图是在参数T = 2 c = 10 h = 5 b = 40 K = 0 alpha = 0.9 Dmean = 3 Dmax = 6 Zmax = 12作的，由于没有订单固定成本，为base-stock policy
Policy Structure2图是在参数T = 2 c = 10 h = 5 b = 40 K = 10 alpha = 0.9 Dmean = 3 Dmax = 6 Zmax = 12作的，为(s, S) policy
"""

import numpy as np
import pandas as pd
import time
from scipy.stats import poisson
import matplotlib.pyplot as plt  

time0 = time.perf_counter()

#初始化全局参数
T = 2  #周期数
c = 10 #订货单价
h = 5 #库存持有成本
b = 40 #补货单价
K = 0 #三级仓库的订单固定成本
alpha = 0.9 #折扣因子
Dmean = 3 #需求的均值
Dmax = 6 #需求的上限
Zmax = 12 #为程序运行时三级仓库的订单上限

#首先生成需求序列，0到10的序列，以及对应的概率值
demand = np.arange(Dmax + 1)
p = poisson.pmf(demand, Dmean)

#初始化V表中的列数据，存储V值的表的大小与系统状态变量的总数有关，T+1期的V值假设为0
X3 = []
X2 = []
X1 = []
for i in range(-Dmax * T, Zmax * T + 1):
    for j in range(-Dmax * T, Zmax * T + 1):
        for k in range(-Dmax * T, Zmax * T + 1):
            X3.append(i)
            X2.append(j)
            X1.append(k)
V = pd.DataFrame({'x3': X3,'x2': X2,'x1':X1})
for t in range(T + 1, 0, -1):
    V[str(t)] = np.zeros(len(X1))


#初始化记录最优决策的表
optimal_action = pd.DataFrame(columns=['t','x1','x2','x3','z','q2','q1'])


#循环开始
#x3\x2\x1代表三个echelon的期初库存
for t in range(T, 0, -1):
    #print('t='+str(t))
    #初始化存放V值的list
    V_t = np.zeros(len(X1))
    #取用上一期的V值
    V_t_plus_1 = V[str(t + 1)]
    for x3 in range(-Dmax * (t - 1), Zmax * (t - 1) + 1):#一直不订货且需求最大，一直订货且下游不订货无需求 
        for x2 in range(-Dmax * (t - 1), x3 + 1):#一直不订货且需求最大，总将第三级仓库的余量订空且无需求
            for x1 in range(-Dmax * (t - 1), x2 + 1):#一直不订货且需求最大，总将第二级仓库的余量订空且无需求
                z_list = []
                q2_list = []
                q1_list = []
                action_cost_list = []
                for z in range(0, Zmax + 1):#向外部订货量是从0到最大订货量
                    for q2 in range(0, x3 - x2 + z + 1):#向三级仓库订货量从0到其当前库存上限
                        for q1 in range(0, x2 - x1 + q2 + 1):#向二级仓库订货量从0到其当前库存上限
                            Lt_and_Future = []
                            for d in demand:
                                index = (x3 + z - d + Dmax * T) * pow(Dmax * T + Zmax * T + 1, 2) + (x2 + q2 - d + Dmax * T) * (Dmax * T + Zmax * T + 1) + (x1 + q1 -d + Dmax * T)
                                Lt_and_Future.append(h * max(x1 + q1 - d, 0) + b * max(d - x1 - q1, 0) + alpha * V_t_plus_1[index])
                            #记录action
                            if z == 0:
                                g = c * (z + q1 + q2) + h * (x3 - x2 + z - q2) + h * (x2 - x1 + q2 - q1) + np.dot(p, Lt_and_Future)
                            else:
                                g = K + c * (z + q1 + q2) + h * (x3 - x2 + z - q2) + h * (x2 - x1 + q2 - q1) + np.dot(p, Lt_and_Future)
                            z_list.append(z)
                            q2_list.append(q2)
                            q1_list.append(q1)
                            action_cost_list.append(g)
                #求出最小成本的action
                min_action_cost = np.min(action_cost_list)
                action_index = np.argmin(action_cost_list)
                V_t[(x3 + Dmax * T) * pow(Dmax * T + Zmax * T + 1, 2) + (x2 + Dmax * T) * (Dmax * T + Zmax * T + 1) + (x1 + Dmax * T)] = min_action_cost
                optimal_action.loc[len(optimal_action['t'])] = [t, x1, x2, x3, z_list[action_index], q2_list[action_index], q1_list[action_index]]
                # print(str(x3)+' '+str(x2)+' '+str(x1))
    V[str(t)] = V_t
                
print(time.perf_counter()-time0)              

#作图             
x = optimal_action[optimal_action['t'] == T]['x3'].tolist()
# x = np.array(optimal_action[optimal_action['t'] == T]['x3'].tolist()) - np.array(optimal_action[optimal_action['t'] == T]['x2'].tolist())
y = optimal_action[optimal_action['t'] == T]['z'].tolist()
          
plt.plot(x, y)
plt.title('Policy Structure')
plt.xlabel('Echelon 3 Inventory Level')
plt.ylabel('Order Quantity')
plt.show()           

# plt.scatter(x, y)
# plt.title('Echelon base-stock level is independent of inventory level')
# plt.xlabel('3rd Inventory')
# plt.ylabel('Order Quantity')
# plt.show()                
                
                
                