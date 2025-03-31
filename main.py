# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:23:11 2025

@author: Sherwin Wu
"""
import numpy as np
import function as fc
import NSGA_II as NSGA
import MOPSO as MOPSO
import MODEA as MODEA
import SPEA2 as SPEA
import random
import time
import matplotlib.pyplot as plt

#%%

def sub_drawGantt(timelist):
    T = timelist.copy()
    #print(T)
    #创建新的图形
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(10,6))
    #每一个工件代表的颜色
    color_map = {}
    for machine in T:
        for task_data in machine[1:]:
            job_idx, operation_idx = int(task_data[1]),int(task_data[2])
            if job_idx not in color_map:
                color_map[job_idx] = (random.random(),random.random(),random.random())
    #print(color_map)
    #遍历机器
    for machine_idx,machine_schedule in enumerate(T):
        for task_data in machine_schedule[1:]:
            start_time,lots_idx,job_idx,end_time = task_data
            color = color_map[lots_idx]
            # 绘制甘特图
            ax.barh(machine_idx,end_time-start_time,left=start_time,height=0.4,color=color)
        #标注
            label = f'{lots_idx}-{job_idx}'
            ax.text((start_time+end_time)/2,machine_idx,label,ha='center',va='center',color='black',fontsize=10)
            #print(task_data)
        
    ax.set_yticks(range(len(T)))
    ax.set_yticklabels([f'line{i+1}' for i in range(len(T))])
    plt.xlabel("Time")
    plt.title("DHHFSP Gantt Chart")
 
    l = []
    sorted_d = dict(sorted(color_map.items(), key=lambda x: x[0]))
    for lots_idx,color in sorted_d.items():
        l.append(plt.Rectangle((0,0),1,1,color=color,label=f'{lots_idx}'))
    plt.legend(handles=l,title='Order')

def pareto_fig(objs,rank):
    objs[:,0] = -objs[:,0]
    pf = objs[np.where(rank == 1)]
    plt.figure()
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    plt.scatter(x, y) 
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('The Pareto front of LSTM_DHFP')
    plt.savefig('Pareto front')
    plt.show()

def calculate_mid(objectives, maximize_indices=[0]):
    """
    计算 Mean Ideal Distance (MID)，支持最小化 & 最大化目标
    :param objectives: (N × m) 的 NumPy 数组，每行是一个解，每列是一个目标
    :param maximize_indices: 列表，表示需要最大化的目标的索引（从 0 开始）
    :return: MID 值
    """
    # 复制数据，避免修改原始数据
    objectives = objectives.copy()

    # 处理最大化目标（取负数）
    objectives[:, maximize_indices] *= -1

    # 计算理想点 (Ideal Point) - 取所有目标的最小值
    ideal_point = np.min(objectives, axis=0)

    # 计算所有解到理想点的欧几里得距离
    distances = np.linalg.norm(objectives - ideal_point, axis=1)

    # 计算 Mean Ideal Distance (MID)
    mid = np.mean(distances)

    return mid

def calculate_sns(pareto_solutions):
    """
    计算 SNS (Spread of Non-dominated Solutions)
    :param pareto_solutions: Pareto 前沿解集 (N × m 数组, N 是解的个数, m 是目标的个数)
    :return: SNS 值
    """
    N = len(pareto_solutions)
    
    if N < 2:
        return 0  # 只有一个解时，SNS 没有意义

    # 按照第一个目标值排序
    pareto_solutions = pareto_solutions[np.argsort(pareto_solutions[:, 0])]

    # 计算极端解间距 df（最左端解和最右端解的欧几里得距离）
    df = np.linalg.norm(pareto_solutions[0] - pareto_solutions[-1])

    # 计算相邻解之间的欧几里得距离
    distances = np.linalg.norm(pareto_solutions[1:] - pareto_solutions[:-1], axis=1)

    # 计算平均距离
    mean_distance = np.mean(distances)

    # 计算 SNS 公式
    sns = (df + np.sum(np.abs(distances - mean_distance))) / (df + (N - 1) * mean_distance)
    
    return sns

def calculate_ras(objectives, maximize_indices=[0]):
    """
    计算 RAS (The Rate of Achievement to Two Objectives Simultaneously)
    :param objectives: (N × 2) NumPy 数组，每行是一个解，每列是一个目标
    :param maximize_indices: 需要最大化的目标索引列表
    :return: RAS 值
    """
    objectives = objectives.copy()  # 避免修改原数据
    N, m = objectives.shape

    # 计算最优和最差值
    best_values = np.min(objectives, axis=0)  # 最小化目标
    worst_values = np.max(objectives, axis=0) # 最小化目标

    # 处理最大化目标（交换 best 和 worst）
    for idx in maximize_indices:
        best_values[idx], worst_values[idx] = worst_values[idx], best_values[idx]

    # 归一化
    normalized = (objectives - best_values) / (worst_values - best_values)

    # 计算每个解的 RAS 值
    ras_values = np.max(normalized, axis=1)

    # 计算平均 RAS
    ras = np.mean(ras_values)

    return ras



#%% NSGA-ii
#pop = fc.creation_pop(76, 100, 5)
#start1 = time.time()
#process_metrx = fc.get_fac_order()
#final_pop,objs,rank = NSGA.main(pop, 20)


#rep,rep_obj = MOPSO.main(pop, 10)
#rep, rep_obj,final_pop = MOPSO.main(pop, 100)
#pf, final_pop = MODEA.main(pop, 100)
#arch_objs,arch_F,final_pop = SPEA.main(pop, 100)

# final_objs = np.array([fc.cal_obj(pop[i]) for i in range(len(final_pop))])  # objectives
#mid_end = time.time()
#record1 = mid_end-start1

# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(m)
#     start1 = time.time()
#     pop = fc.creation_pop(76, 50, 5)
#     mid_final_pop,mid_objs,mid_rank = NSGA.main(pop, 10)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     mid = calculate_mid(mid_objs,0)
#     sns = calculate_sns(mid_objs)
#     ras = calculate_ras(mid_objs)
#     fial_pop_record.append(mid_final_pop)
#     objs_record.append(mid_objs)
#     time_record.append(record1)
#     rank_record.append(mid_rank)
#     pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])

#%%  MOPSO
# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(loop)
#     start1 = time.time()
#     pop = fc.creation_pop(76, 50, 5)
#     _, _,final_pop = MOPSO.main(pop, 50)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     rep_obj = objs = np.array([fc.cal_obj(final_pop[i]) for i in range(len(final_pop))])  # objectives
#     rep_obj[:,0] = -rep_obj[:,0]
#     #rep_obj[:,1] = -rep_obj[:,1] 
#     mid = calculate_mid(rep_obj)
#     sns = calculate_sns(rep_obj)
#     ras = calculate_ras(rep_obj)
#     fial_pop_record.append(final_pop)
#     objs_record.append(rep_obj)
#     time_record.append(record1)
#     #rank_record.append(rep)
#     #pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])
#     plt.figure()
#     x = [o[0] for o in rep_obj]
#     y = [o[1] for o in rep_obj]
#     plt.scatter(x, y)
#     plt.show()


#%%  MODEA
# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(m)
#     start1 = time.time()
#     pop = fc.creation_pop(76, 50, 5)
#     _, final_pop = MODEA.main(pop, 50)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     rep_obj = objs = np.array([fc.cal_obj(final_pop[i]) for i in range(len(final_pop))])  # objectives
#     rep_obj[:,0] = -rep_obj[:,0]
#     #rep_obj[:,1] = -rep_obj[:,1] 
#     mid = calculate_mid(rep_obj)
#     sns = calculate_sns(rep_obj)
#     ras = calculate_ras(rep_obj)
#     fial_pop_record.append(final_pop)
#     objs_record.append(rep_obj)
#     time_record.append(record1)
#     #rank_record.append(rep)
#     #pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])
#     plt.figure()
#     x = [o[0] for o in rep_obj]
#     y = [o[1] for o in rep_obj]
#     plt.scatter(x, y)
#     plt.show()
    
#%% spea2
# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(m)
#     start1 = time.time()
#     pop = fc.creation_pop(76, 50, 5)
#     _,_,final_pop = SPEA.main(pop, 50)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     rep_obj = objs = np.array([fc.cal_obj(final_pop[i]) for i in range(len(final_pop))])  # objectives
#     rep_obj[:,0] = -rep_obj[:,0]
#     #rep_obj[:,1] = -rep_obj[:,1] 
#     mid = calculate_mid(rep_obj)
#     sns = calculate_sns(rep_obj)
#     ras = calculate_ras(rep_obj)
#     fial_pop_record.append(final_pop)
#     objs_record.append(rep_obj)
#     time_record.append(record1)
#     #rank_record.append(rep)
#     #pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])
#     plt.figure()
#     x = [o[0] for o in rep_obj]
#     y = [o[1] for o in rep_obj]
#     plt.scatter(x, y)
#     plt.show()

#%% NSGA-ii heuristic
#pop = fc.creation_pop(76, 100, 5)
#start1 = time.time()
#process_metrx = fc.get_fac_order()
#final_pop,objs,rank = NSGA.main(pop, 20)


#rep,rep_obj = MOPSO.main(pop, 10)
#rep, rep_obj,final_pop = MOPSO.main(pop, 100)
#pf, final_pop = MODEA.main(pop, 100)
#arch_objs,arch_F,final_pop = SPEA.main(pop, 100)

# final_objs = np.array([fc.cal_obj(pop[i]) for i in range(len(final_pop))])  # objectives
#mid_end = time.time()
#record1 = mid_end-start1

loop = 1
fial_pop_record = []
objs_record = []
rank_record = []
time_record=[]
metrix = []
for m in range(loop):
    print(m)
    pop = fc.heuristic_creation_pop(76, 50, 5)
    start1 = time.time()
    mid_final_pop,mid_objs,mid_rank = NSGA.main(pop, 50)
    mid_end = time.time()
    record1 = mid_end-start1
    mid = calculate_mid(mid_objs,0)
    sns = calculate_sns(mid_objs)
    ras = calculate_ras(mid_objs)
    fial_pop_record.append(mid_final_pop)
    objs_record.append(mid_objs)
    time_record.append(record1)
    rank_record.append(mid_rank)
    pareto_fig(mid_objs,mid_rank)
    metrix.append([mid,sns,ras])

#%%  MOPSO heuristic
# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(m)
#     pop = fc.heuristic_creation_pop(76, 50, 5)
#     start1 = time.time()
#     _, _,final_pop = MOPSO.main(pop, 50)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     rep_obj = objs = np.array([fc.cal_obj(final_pop[i]) for i in range(len(final_pop))])  # objectives
#     rep_obj[:,0] = -rep_obj[:,0]
#     #rep_obj[:,1] = -rep_obj[:,1] 
#     mid = calculate_mid(rep_obj)
#     sns = calculate_sns(rep_obj)
#     ras = calculate_ras(rep_obj)
#     fial_pop_record.append(final_pop)
#     objs_record.append(rep_obj)
#     time_record.append(record1)
#     #rank_record.append(rep)
#     #pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])
#     plt.figure()
#     x = [o[0] for o in rep_obj]
#     y = [o[1] for o in rep_obj]
#     plt.scatter(x, y)
#     plt.show()


#%%  MODEA heuristic
# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(m)
#     pop = fc.heuristic_creation_pop(76, 50, 5) 
#     start1 = time.time()
#     _, final_pop = MODEA.main(pop, 50)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     rep_obj = objs = np.array([fc.cal_obj(final_pop[i]) for i in range(len(final_pop))])  # objectives
#     rep_obj[:,0] = -rep_obj[:,0]
#     #rep_obj[:,1] = -rep_obj[:,1] 
#     mid = calculate_mid(rep_obj)
#     sns = calculate_sns(rep_obj)
#     ras = calculate_ras(rep_obj)
#     fial_pop_record.append(final_pop)
#     objs_record.append(rep_obj)
#     time_record.append(record1)
#     #rank_record.append(rep)
#     #pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])
#     plt.figure()
#     x = [o[0] for o in rep_obj]
#     y = [o[1] for o in rep_obj]
#     plt.scatter(x, y)
#     plt.show()

#%% spea2 heuristic
# loop = 10
# fial_pop_record = []
# objs_record = []
# rank_record = []
# time_record=[]
# metrix = []
# for m in range(loop):
#     print(m)
#     pop = fc.heuristic_creation_pop(76, 50, 5)
#     start1 = time.time()
#     _,_,final_pop = SPEA.main(pop, 50)
#     mid_end = time.time()
#     record1 = mid_end-start1
#     rep_obj = objs = np.array([fc.cal_obj(final_pop[i]) for i in range(len(final_pop))])  # objectives
#     rep_obj[:,0] = -rep_obj[:,0]
#     #rep_obj[:,1] = -rep_obj[:,1] 
#     mid = calculate_mid(rep_obj)
#     sns = calculate_sns(rep_obj)
#     ras = calculate_ras(rep_obj)
#     fial_pop_record.append(final_pop)
#     objs_record.append(rep_obj)
#     time_record.append(record1)
#     #rank_record.append(rep)
#     #pareto_fig(mid_objs,mid_rank)
#     metrix.append([mid,sns,ras])
#     plt.figure()
#     x = [o[0] for o in rep_obj]
#     y = [o[1] for o in rep_obj]
#     plt.scatter(x, y)
#     plt.show()

#rep_obj= np.array([fc.cal_obj(mid_final_pop[i]) for i in range(len(mid_final_pop))])
# single_pop = fial_pop_record[0][5]
# sinobj = fc.cal_obj(single_pop)
# conver_s_pop = fc.chrome_inverse_singel(single_pop)
# process_metrix = fc.get_fac_order()
# T,C = fc.getschedule(conver_s_pop, process_metrix, 5)
