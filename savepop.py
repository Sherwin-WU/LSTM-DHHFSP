# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:58:20 2025

@author: zhangfn
"""

import pandas as pd
import numpy as np

# 假设你的数据已经以某种形式存在，这里我们手动创建一些示例数据
# 注意：实际使用时，你需要根据你的数据格式来创建这些数据
data = []
for i in range(len(final_pop)):  # 假设有100个数据项
    # 这里我们创建了一个2x76的随机浮点数数组作为示例
    array = final_pop[i]
    # 将数组添加到字典中，并使用索引作为键（或者你可以使用其他描述性键）
    data.append({"index_" + str(i): array})

# 创建DataFrame
df = pd.DataFrame(data)

# 导出为JSON文件
df.to_json('SPEA_pop_data.json', orient='records', lines=True)

print("数据已成功导出到data.json文件")





# 读取 JSON 文件
df_loaded = pd.read_json('data.json', lines=True)

# 转换为字典并还原 NumPy 数组
restored_data = {}
for idx, row in df_loaded.iterrows():
    key = f"index_{idx}"
    restored_data[key] = np.array(row[key])  # 从每行提取键值对并转换‌:ml-citation{ref="6" data="citationList"}

final_pop1 =[]
for i in range(len(restored_data)):
    key = f"index_{i}"
    now_array = restored_data[key]
    final_pop1.append(now_array)


# 验证数据
sample_key = "index_0"
print(f"样本 {sample_key} 形状:", restored_data[sample_key].shape)
print("前 5 个元素:", restored_data[sample_key][:5])
