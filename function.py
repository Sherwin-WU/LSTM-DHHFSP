# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 13:59:29 2025

@author: Sherwin Wu
"""
import pandas as pd
import random
import numpy as np
import copy
def get_fac_order():
    df = pd.read_excel('dataset1.xlsx', sheet_name='Sheet4')
    a,b = np.shape(df)
    process_metrix = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            process_metrix[i][0] = df['sale_order'][i]
            process_metrix[i][1] = df['task_id'][i]
            process_metrix[i][2] = df['line1'][i]
            process_metrix[i][3] = df['line2'][i]
            process_metrix[i][4] = df['line3'][i]
            process_metrix[i][5] = df['line4'][i]
            process_metrix[i][6] = df['line5'][i]
    return process_metrix

# process_metrix = get_fac_order()
#%% 生成原始染色体
def compare_fac(T):
    fac_num = len(T)
    time_record = np.zeros((fac_num,2),dtype = float)
    for i in range(fac_num):
        time_record[i] = [i,T[i][-1][-1]]
    sorted_time_record = time_record[time_record[:, 1].argsort()]
    fac_id = sorted_time_record[0][0]
    return fac_id


def heuristic_create_code(lot_num,fac_num):
    fac_num = 5
    process_matrix = get_fac_order()
    max_order_num = int(process_matrix[-1][0])
    lot_num = int(process_matrix[-1][1])+1
    code = np.zeros((2,lot_num),dtype = float)
    random_sequence = random.sample(range(max_order_num+1), max_order_num+1)
    T = [[[0]] for _ in range(fac_num)]
    C = np.zeros([lot_num,3],dtype=int)
    k=0
    for i in random_sequence:
        now_tasks  = process_matrix[np.where(process_matrix[:,0] == i)]
        np.random.shuffle(now_tasks)
        for j in range(len(now_tasks)):
            now_task_id = int(now_tasks[j][1])
            now_fac_id = int(compare_fac(T))
            now_sale_order_id = process_matrix[now_task_id][0]
            now_process_time = process_matrix[now_task_id][now_fac_id+2]
            last_complete_time = T[now_fac_id][-1][-1] if len(T[now_fac_id])>0 else 0
            now_task_start_time = last_complete_time + random.randrange(1, 5)
            now_task_complete_time = now_task_start_time + now_process_time 
            insert_index = len(T[now_fac_id])
            T[now_fac_id].insert(insert_index,[now_task_start_time,now_sale_order_id,now_task_id,now_task_complete_time])
            C[now_task_id] = [now_sale_order_id,now_task_id,now_task_complete_time]
            code[0][k] = now_task_id
            code[1][k] = now_fac_id
            k+=1
    I = [random.uniform(0, 1) for _ in range(lot_num)]
    sorted_I = sorted(I)
    for m in range(lot_num):
        copyed_code = copy.deepcopy(code)
        first_layer = code[0,:]
        code[0][m] = sorted_I[int(first_layer[m])]
    return code

def creation_code(lot_num,fac_num):
     code = np.zeros((2,lot_num),dtype = float)
     L = np.zeros((1,lot_num),dtype =int)
     I = [random.uniform(0, 1) for _ in range(lot_num)]
     random.shuffle(I)
     for i in range(lot_num):
         now_fac = random.randint(0,fac_num-1)
         L[0][i] = now_fac
     code[0]=I
     code[1] = L
     return code
def creation_pop(lot_num,popsize,fac_num): #生成种群
    chromes=[]
    for i in range(popsize):
        singel_pop=creation_code(lot_num,fac_num)
        chromes.append(singel_pop)
    return chromes
def heuristic_creation_pop(lot_num,popsize,fac_num):
    chromes=[]
    for i in range(int(popsize/2)):
        singel_pop=heuristic_create_code(lot_num,fac_num)
        chromes.append(singel_pop)
    for j in range(int(popsize/2),popsize):
        singel_pop=creation_code(lot_num,fac_num)
        chromes.append(singel_pop)
    return chromes
def find_chrome_layer1_index(nowrank,layer1):
    for i in range(len(layer1)):
        if nowrank == layer1[i]:
            return i
def chrome_inverse_all(pop):
    new_pop = []
    for i in range(len(pop)):
        mid_pop  = np.zeros((2,len(pop[0][1])),dtype = float)
        now_pop_layer1 = pop[i][0]
        now_pop_layer2 = pop[i][1]
        sort_layer1 = sorted(now_pop_layer1)
        new_sort_layer1 = np.zeros((1,len(pop[0][1])),dtype = float)
        for j in range(len(now_pop_layer1)):
            now_idx = find_chrome_layer1_index(sort_layer1[j],now_pop_layer1)
            new_sort_layer1[0][j] = now_idx
        mid_pop[0] = new_sort_layer1
        mid_pop[1] = now_pop_layer2
        new_pop.append(mid_pop)
    return new_pop
        
def chrome_inverse_singel(pop):
    mid_pop  = np.zeros((2,len(pop[0])),dtype = float)
    now_pop_layer1 = pop[0]
    now_pop_layer2 = pop[1]
    sort_layer1 = sorted(now_pop_layer1)
    new_sort_layer1 = np.zeros((1,len(pop[0])),dtype = float)
    for j in range(len(now_pop_layer1)):
        now_idx = find_chrome_layer1_index(sort_layer1[j],now_pop_layer1)
        new_sort_layer1[0][j] = now_idx
    mid_pop[0] = new_sort_layer1
    mid_pop[1] = now_pop_layer2
    return mid_pop
#pop=creation_pop(10,5,5)
# new_pop = chrome_inverse(pop)

def code_process(singlechrome,fac_num): #将编码好的染色体分成工厂形式的染色体
    lots_num = len(singlechrome[1])
    fac_lots = []
    for i in range(fac_num):
        sig_chrome = []
        k=0
        for j in range(lots_num):
            if singlechrome[1][j] == i:
                sig_chrome.append(int(singlechrome[0][j]))
                k +=1
        fac_lots.append(sig_chrome)
    return fac_lots

def getschedule(singlechrome,process_metrix,fac_num):
    fac_lots = code_process(singlechrome, fac_num)
    task_num = len(process_metrix)
    T = [[[0]] for _ in range(fac_num)]
    C = np.zeros([task_num,3],dtype=int)
    for i in range(len(fac_lots)):
        now_pro_task = fac_lots[i]
        for j in range(len(now_pro_task)):
            now_task_id = now_pro_task[j]
            now_sale_order_id = process_metrix[now_task_id][0]
            now_process_time = process_metrix[now_task_id][i+2]
            last_complete_time = T[i][-1][-1] if len(T[i])>0 else 0
            now_task_start_time = last_complete_time + random.randrange(1, 5)
            now_task_complete_time = now_task_start_time + now_process_time 
            insert_index = len(T[i])
            T[i].insert(insert_index,[now_task_start_time,now_sale_order_id,now_task_id,now_task_complete_time])
            C[now_task_id] = [now_sale_order_id,now_task_id,now_task_complete_time]
    return T,C
# fac_lots = code_process(new_pop[0], 5)
# T,C = getschedule(new_pop[0], process_metrix, 5)

def function1(T,C):
    max_complete_line=[]
    for i in range(len(T)):
        now_max_com = T[i][-1][-1]
        max_complete_line.append(now_max_com)
    fac1_max = max(max_complete_line[0],max_complete_line[1])
    fac1_balance = (max_complete_line[0]+max_complete_line[1])/(2*fac1_max)
    fac2_max = max(max_complete_line[2],max_complete_line[3],max_complete_line[4])
    fac2_balance = (max_complete_line[2]+max_complete_line[3]+max_complete_line[4])/(3*fac2_max)
    total_balance = -(fac1_balance+fac2_balance)/2
    return total_balance

def function2(T,C):
    max_order = 23 # set as my order number
    order_record = []
    for i in range(max_order):
        #now_order_time=np.zeros((0,3))
        now_order_time = C[np.where(C[:,0] == i)]
        if len(now_order_time)>1 :
            now_order_max = max(now_order_time[:,2])
            mow_order_min = min(now_order_time[:,2])
            order_record.append([now_order_max-mow_order_min])
    total_value=0
    for k in range(len(order_record)):
        total_value = total_value + order_record[k][0]
    ave_value = total_value/max_order
    return ave_value


def z_score_normalize(data):
    """
    使用Z-Score标准化对数据进行归一化，使数据具有均值为0和标准差为1。

    参数:
    data (numpy.ndarray): 需要归一化的数据，可以是一维数组或多维数组。

    返回:
    numpy.ndarray: 归一化后的数据，形状与输入数据相同。
    """
    # 将数据转换为NumPy数组（如果它还不是的话）
    data = np.asarray(data)
    
    # 计算数据的均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)  # ddof=1表示计算样本标准差
    
    # 防止除以零的情况
    std[std == 0] = 1
    
    # 应用Z-Score标准化公式进行归一化
    normalized_data = (data - mean) / std
    
    return normalized_data

def min_max_normalize(data):
    """
    对数据进行最小-最大归一化，将数据缩放到[new_min, new_max]区间。

    参数:
    data (numpy.ndarray): 需要归一化的数据，可以是一维数组或多维数组。
    new_min (float): 归一化后数据的最小值，默认为0。
    new_max (float): 归一化后数据的最大值，默认为1。

    返回:
    numpy.ndarray: 归一化后的数据，形状与输入数据相同。
    """
    # 将数据转换为NumPy数组（如果它还不是的话）
    data = np.asarray(data, dtype=float)
    new_min=0
    new_max=1
    
    # 计算数据的最小值和最大值
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    # 防止除以零（虽然理论上不应该出现，但为了代码的健壮性）
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    # 应用最小-最大归一化公式
    normalized_data = (data - min_val) / range_val * (new_max - new_min) + new_min
    
    return normalized_data


def cal_obj(singlepop):
    new_pop = chrome_inverse_singel(singlepop)
    process_metrix = get_fac_order()
    T,C = getschedule(new_pop, process_metrix, 5)
    f1 = function1(T, C)
    f2 = function2(T, C)
    return [f1, f2]
#total_balance = function1(T,C)
#ave_value = function2(T, C)
