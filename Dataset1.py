# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:06:49 2025

@author: zhangfn
"""
import pandas as pd
import random
import numpy as np
def match_code(Routing_code_list,now_code):
    now_routing = len(Routing_code_list)
    for i in range(now_routing):
        if Routing_code_list[i][1] == now_code:
            return True
        elif i==now_routing-1:
            return False
def match_order(ORDER_list,now_order):
    now_order_num = len(ORDER_list)
    for i in range(now_order_num):
        if ORDER_list[i][1] == now_order:
            return True
        elif i==now_order_num-1:
            return False

def find_match_code(Routing_code_list,now_code):
    for i in range(len(Routing_code_list)):
        if Routing_code_list[i][1] == now_code:
            return i
        
def find_max_length(ini_data):
    last_data = ini_data[-1][0]
    data_calcu=[]
    for i in range(last_data):
        mid_data = sum(1 for row in ini_data if i in row[:1])
        data_calcu.append(mid_data)
    max_cum = max(data_calcu)
    return last_data,max_cum
    

# def get_fac_order():
#     df = pd.read_excel('dataset1.xlsx', sheet_name='Sheet1')
#     Total_len = len(df)
#     Routing_code_list=[]
#     ORDER_list=[]
#     for i in range(Total_len):
#         now_cate = df['ROUTING_CODE'][i]
#         now_routing = len(Routing_code_list)
#         if now_routing == 0:
#             Routing_code_list.append([now_routing+1,now_cate])
#         else:
#             verify = match_code(Routing_code_list, now_cate)
#             if verify == False:
#                 Routing_code_list.append([now_routing+1,now_cate])
#     order_idx=0
#     for i in range(Total_len):
#         now_order = df['SALE_ORDER_ID'][i]
#         now_order_ID = len(ORDER_list)
#         if now_order_ID == 0:
#             ORDER_list.append([order_idx,now_order])
#             order_idx+=1
#         else:
#             verify = match_order(ORDER_list,now_order)
#             if verify == False:
#                 ORDER_list.append([order_idx,now_order])
#                 order_idx+=1
#     ini_data=[]
#     now_input_idx=0
#     for i in range(Total_len):
#         now_cate = df['ROUTING_CODE'][i]
#         now_order = df['SALE_ORDER_ID'][i]
#         if i==0:
#             process_time = random.uniform(1, 3)
#             now_code_idx = find_match_code(Routing_code_list, now_cate)
#             ini_data.append([now_input_idx,now_code_idx,process_time])
#         else:
#             now_code_idx = find_match_code(Routing_code_list, now_cate)
#             if (df['ROUTING_CODE'][i] == df['ROUTING_CODE'][i-1] and df['SALE_ORDER_ID'][i] == df['SALE_ORDER_ID'][i-1]):
#                 process_time = process_time+random.uniform(1, 3)
#                 ini_data.append([now_input_idx,now_code_idx,process_time])
#             else:
#                 now_input_idx+=1
#                 process_time = random.uniform(1, 3)
#                 ini_data.append([now_input_idx,now_code_idx,process_time])
#     return ini_data

# df = pd.read_excel('dataset1.xlsx', sheet_name='Sheet2')
# Total_len = len(df)
# Routing_code_list=[]
# ORDER_list=[]
# for i in range(Total_len):
#     now_cate = df['ROUTING_CODE'][i]
#     now_routing = len(Routing_code_list)
#     if now_routing == 0:
#         Routing_code_list.append([now_routing+1,now_cate])
#     else:
#         verify = match_code(Routing_code_list, now_cate)
#         if verify == False:
#             Routing_code_list.append([now_routing+1,now_cate])
# order_idx=0
# for i in range(Total_len):
#     now_order = df['SALE_ORDER_ID'][i]
#     now_order_ID = len(ORDER_list)
#     if now_order_ID == 0:
#         ORDER_list.append([order_idx,now_order])
#         order_idx+=1
#     else:
#         verify = match_order(ORDER_list,now_order)
#         if verify == False:
#             ORDER_list.append([order_idx,now_order])
#             order_idx+=1
# ini_data=[]
# now_input_idx=0
# prcess1 = 15
# process2 = 5
# process3 = 6
# process4 = 5
# process5 = 15
# process_mtrix = np.zeros((1, 5))
# for i in range(Total_len):
#     now_cate = df['ROUTING_CODE'][i]
#     now_order = df['SALE_ORDER_ID'][i]
#     if i==0:
#         mid_prod1 = prcess1 
#         mid_prod2 = mid_prod1+process2
#         mid_prod3 = mid_prod2+process3
#         mid_prod4 = mid_prod3+process4
#         mid_prod5 = mid_prod4+process5
#         now_code_idx = find_match_code(Routing_code_list, now_cate)
#         process_mtrix[0][0] = mid_prod1
#         process_mtrix[0][1] = mid_prod2
#         process_mtrix[0][2] = mid_prod3
#         process_mtrix[0][3] = mid_prod4
#         process_mtrix[0][4] = mid_prod5
#         ini_data.append([now_input_idx,now_code_idx,process_mtrix[0][-1]])
#     else:
#         now_code_idx = find_match_code(Routing_code_list, now_cate)
#         if (df['ROUTING_CODE'][i] == df['ROUTING_CODE'][i-1] and df['SALE_ORDER_ID'][i] == df['SALE_ORDER_ID'][i-1]):
#             mid_prod1 = process_mtrix[i-1][0]+prcess1
#             mid_prod2 = max(mid_prod1,process_mtrix[i-1][1])+process2
#             mid_prod3 = max(mid_prod2,process_mtrix[i-1][2])+process3
#             mid_prod4 = max(mid_prod3,process_mtrix[i-1][3])+process4
#             mid_prod5 = max(mid_prod4,process_mtrix[i-1][4])+process5
#             new_row = np.array([[mid_prod1, mid_prod2, mid_prod3,mid_prod4,mid_prod5]])
#             process_mtrix = np.concatenate((process_mtrix, new_row), axis=0)
#             ini_data.append([now_input_idx,now_code_idx,process_mtrix[i][-1]])
#         else:
#             now_input_idx+=1
#             mid_prod1 = prcess1 
#             mid_prod2 = mid_prod1+process2
#             mid_prod3 = mid_prod2+process3
#             mid_prod4 = mid_prod3+process4
#             mid_prod5 = mid_prod4+process5
#             new_row = np.array([[mid_prod1, mid_prod2, mid_prod3,mid_prod4,mid_prod5]])
#             process_mtrix = np.concatenate((process_mtrix, new_row), axis=0)
#             ini_data.append([now_input_idx,now_code_idx,process_mtrix[i][-1]])

def get_fac_order():
    df = pd.read_excel('dataset1.xlsx', sheet_name='Sheet2')
    Total_len = len(df)
    Total_len = 10000
    Routing_code_list=[]
    ORDER_list=[]
    for i in range(Total_len):
        now_cate = df['ROUTING_CODE'][i]
        now_routing = len(Routing_code_list)
        if now_routing == 0:
            Routing_code_list.append([now_routing+1,now_cate])
        else:
            verify = match_code(Routing_code_list, now_cate)
            if verify == False:
                Routing_code_list.append([now_routing+1,now_cate])
    order_idx=0
    for i in range(Total_len):
        now_order = df['SALE_ORDER_ID'][i]
        now_order_ID = len(ORDER_list)
        if now_order_ID == 0:
            ORDER_list.append([order_idx,now_order])
            order_idx+=1
        else:
            verify = match_order(ORDER_list,now_order)
            if verify == False:
                ORDER_list.append([order_idx,now_order])
                order_idx+=1
    ini_data=[]
    now_input_idx=0
    prcess1 = random.randrange(15, 20)
    process2 = random.randrange(5, 10)
    process3 = random.randrange(6, 10)
    process4 = random.randrange(5, 10)
    process5 = random.randrange(10, 20)
    process_mtrix = np.zeros((1, 5))
    for i in range(Total_len):
        now_cate = df['ROUTING_CODE'][i]
        now_order = df['SALE_ORDER_ID'][i]
        if i==0:
            mid_prod1 = prcess1 
            mid_prod2 = mid_prod1+process2
            mid_prod3 = mid_prod2+process3
            mid_prod4 = mid_prod3+process4
            mid_prod5 = mid_prod4+process5
            now_code_idx = find_match_code(Routing_code_list, now_cate)
            process_mtrix[0][0] = mid_prod1
            process_mtrix[0][1] = mid_prod2
            process_mtrix[0][2] = mid_prod3
            process_mtrix[0][3] = mid_prod4
            process_mtrix[0][4] = mid_prod5
            ini_data.append([now_input_idx,now_code_idx,process_mtrix[0][-1]])
        else:
            now_code_idx = find_match_code(Routing_code_list, now_cate)
            if (df['ROUTING_CODE'][i] == df['ROUTING_CODE'][i-1] and df['SALE_ORDER_ID'][i] == df['SALE_ORDER_ID'][i-1]):
                mid_prod1 = process_mtrix[i-1][0]+prcess1
                mid_prod2 = max(mid_prod1,process_mtrix[i-1][1])+process2
                mid_prod3 = max(mid_prod2,process_mtrix[i-1][2])+process3
                mid_prod4 = max(mid_prod3,process_mtrix[i-1][3])+process4
                mid_prod5 = max(mid_prod4,process_mtrix[i-1][4])+process5
                new_row = np.array([[mid_prod1, mid_prod2, mid_prod3,mid_prod4,mid_prod5]])
                process_mtrix = np.concatenate((process_mtrix, new_row), axis=0)
                ini_data.append([now_input_idx,now_code_idx,process_mtrix[i][-1]])
            else:
                now_input_idx+=1
                mid_prod1 = prcess1 
                mid_prod2 = mid_prod1+process2
                mid_prod3 = mid_prod2+process3
                mid_prod4 = mid_prod3+process4
                mid_prod5 = mid_prod4+process5
                new_row = np.array([[mid_prod1, mid_prod2, mid_prod3,mid_prod4,mid_prod5]])
                process_mtrix = np.concatenate((process_mtrix, new_row), axis=0)
                ini_data.append([now_input_idx,now_code_idx,process_mtrix[i][-1]])
    return ini_data

