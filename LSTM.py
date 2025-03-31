# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:23:14 2025

@author: zhangfn
"""
#%%
import tensorflow as tf
import numpy as np
import json
import time
import pickle
import matplotlib.pyplot as plt
import Dataset1 as inidata



tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 限制 GPU 内存增长
        tf.config.set_visible_devices(gpus[0], 'GPU')  # 只使用第一个 GPU
        print("GPU 设置成功")
    except RuntimeError as e:
        print(e)
#%%
a = inidata.get_fac_order()
d0,d1 = inidata.find_max_length(a)
# d0 = 56
# d1 = 20

x_typ1 = np.zeros((d0, d1, 1))
for i in range(d0):
    idx=0
    for j in range(len(a)):
        if a[j][0] == i:
            x_typ1[i,idx,0] = 1
            idx+=1
            

x_ftr1 = np.zeros((d0, d1, 1))
for i in range(d0):
    idx=0
    for j in range(len(a)):
        if a[j][0] == i:
            x_ftr1[i,idx,0] = ((a[j][1])+1)/10
            idx+=1


y_loc1 = np.zeros((d0, d1, 1))
for i in range(d0):
    idx=0
    for j in range(len(a)):
        if a[j][0] == i:
            y_loc1[i,idx,0] = a[j][2]
            idx+=1
            
y_mkp = y_loc1

y_lstm = y_mkp

#%%
def shuffle(arr, N, seed=123):
    idx = np.arange(N)
    RD = np.random.RandomState(seed)
    RD.shuffle(idx)
    
    res = []
    for data in arr:
        res.append(data[idx])
    return res

N = len(x_typ1)

[x_ftr1, x_typ1, y_lstm, y_loc1, y_mkp] = shuffle([x_ftr1, x_typ1, y_lstm, y_loc1, y_mkp], N)

N_train = int(N * 0.7)

traindata = [[x_ftr1[:N_train], x_typ1[:N_train]], y_lstm[:N_train]]
testdata = [[x_ftr1[N_train:], x_typ1[N_train:]], y_lstm[N_train:]]

test_loc = y_loc1[N_train:]

#%%
def loss_fn(y_pred, y_true):
    mse = tf.reduce_mean((y_pred[y_pred!=0] - y_true[y_pred!=0])**2)
    return mse

lstm_units = 128
dense_units = 128
reg_l1, reg_l2 = 0.08, 0.1
epochs = 400
batch_size = 32
dropout = 0.2

input1 = tf.keras.Input(shape=(d1, 1), name='node_feature')
mask = tf.keras.Input(shape=(d1, 1), name='node_type')

# First LSTM layer
h1 = tf.keras.layers.LSTM(lstm_units,
                          return_sequences=True,
                          return_state=False,
                          dropout=dropout
                          )(input1)

# Normalize the outputs for LSTM_1
h1 = tf.keras.layers.LayerNormalization(axis=1)(h1)

# Skip-level connection
input2 = tf.keras.layers.Concatenate(axis=-1)([h1, input1])

# Second LSTM layer
h2 = tf.keras.layers.LSTM(lstm_units,
                          return_sequences=True,
                          return_state=False,
                          dropout=dropout
                          )(input2)

# Normalize the outputs for LSTM_2
h2 = tf.keras.layers.LayerNormalization(axis=1)(h2)

# Skip-level connection
input3 = tf.keras.layers.Concatenate(axis=-1)([h1, h2])

# MLP layer 1
x = tf.keras.layers.Dense(dense_units,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l1_l2(reg_l1, reg_l2)
                          )(input3)

# MLP layer 2
x = tf.keras.layers.Dense(dense_units,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l1_l2(reg_l1, reg_l2)
                          )(x)
# MLP layer 3
x = tf.keras.layers.Dense(dense_units,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l1_l2(reg_l1, reg_l2)
                          )(x)

# Output layer
x = tf.keras.layers.Dense(1)(x)

# Mask output for machine units and null units
y = tf.keras.layers.multiply([x, mask])

# Assemble Model
model = tf.keras.Model(inputs=[input1, mask], outputs=y)

#%%
optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, momentum=0.0, epsilon=1e-07, clipvalue=1.)

#%%
tf.random.set_seed(seed=12345)
model.compile(optimizer=optim,
              loss=loss_fn,
              metrics=[tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.RootMeanSquaredError()])

# Early stopping
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_square', patience=20)

# Training
hist = model.fit(x=traindata[0], y=traindata[1], validation_data=testdata, verbose=0, epochs=epochs, batch_size=batch_size) #, callbacks=[callback])

#%%
plt.figure(figsize=[10, 5], dpi=300)
plt.plot(hist.history['loss'], color='red')
plt.plot(hist.history['val_loss'], linestyle=':', color='b')

plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig('Training_process.png')
plt.show()

#%%
# plt.figure()
# plt.plot(hist.history['root_mean_squared_error'])
# plt.plot(hist.history['val_root_mean_squared_error'])

# plt.legend(['train', 'test'])

# plt.show()

#%%
# Predicted Value
y_pred = model.predict(x=testdata[0])
# Mask
y_mask = testdata[0][1]
# Actual Value
y_real = testdata[1]

#%%
def plot_prediction(idx):
    xx = [x for x in range(d1) if testdata[0][1][idx][x] == 1 ]

    yy_pred = (y_pred[idx])[testdata[0][1][idx] == 1]

    #yy_real = (y_real[idx])[testdata[0][1][idx] == 1]

    yy_loc = test_loc[idx][testdata[0][1][idx] == 1]

    #plt.figure(figsize=[10, 5], dpi=400)

    fig, ax = plt.subplots(figsize=(15,7), dpi=200)


    ax.plot(xx, yy_pred, marker='o')
    #ax.plot(xx, yy_real, marker='s')
    ax.plot(xx, yy_loc, marker='d')
    ax.legend(['Predicted PCT', 'Real PCT', 'PCT Bound'], fontsize = 15.0)

    ax.set_ylabel('Time', fontsize = 15.0)
    ax.set_xlabel('Product', fontsize = 15.0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 

    ax.invert_xaxis()

    fig.savefig('prediction.png')
    fig.show()
    
idx = np.random.choice(int(d0*0.7))
plot_prediction(idx)

mae = np.mean(np.abs(y_pred[y_pred!=0] - y_real[y_pred!=0]))
print("Mean Absolute Error: %g" % mae)

mape = np.mean(np.abs((y_real[y_pred!=0] - y_pred[y_pred!=0]) / y_real[y_pred!=0]))
print("Mean Absolute Percentage Error: %g%%" % (mape*100))

mse = np.mean((y_pred[y_pred!=0] - y_real[y_pred!=0]) ** 2)
print("Mean Square Error: %g" % mse)

rmse = np.sqrt(np.mean((y_pred[y_pred!=0] - y_real[y_pred!=0]) ** 2))
print("Root Mean Square Error: %g" % rmse)

r2 = 1 - np.sum((y_real[y_pred!=0] - y_pred[y_pred!=0]) ** 2) / np.sum((np.mean(y_real[y_pred!=0]) - y_real[y_mask!=0]) ** 2)
print("R2: %g" % r2)

predicted = np.squeeze(y_pred[y_pred!=0])
actual = np.squeeze(y_real[y_pred!=0])
location = np.squeeze(test_loc[y_pred!=0])

np.sum(predicted>=location) / predicted.shape[0]

fig, ax = plt.subplots()
ax.scatter(actual, predicted, marker='.', alpha=0.05)
ax.plot([-20, 200], [-20, 200], color='black')
ax.set_xlim([-20, 200])
ax.set_ylim([-20, 200])


plt.show()

fig, ax = plt.subplots(figsize=(10, 6), dpi=500)
ax.scatter([1000], [1000], color='blue', marker='o')
ax.scatter([1000], [1000], color='m', marker='o')
ax.legend(['Realistic Predictions', 'Unrealistic Predictions'], fontsize=20)
ax.scatter(actual[predicted>=location], predicted[predicted>=location], marker='.', alpha=0.3, c='b', s=500)
ax.scatter(actual[predicted<location], predicted[predicted<location], marker='.', alpha=0.3, c='m', s=500)
ax.plot([-20, 800], [-20, 800], color='black')
ax.set_xlim([-20, 800])
ax.set_ylim([-20, 800])
ax.set_title('Pure LSTM', fontsize=20)
ax.set_xlabel('True PCT', fontsize=20)
ax.set_ylabel('Predicted PCT', fontsize=20)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
fig.savefig("pure_lstm.png")
plt.show()



fig, ax = plt.subplots(figsize=(10, 6), dpi=500)

# Plot realistic and unrealistic predictions (for legend purposes)
# ax.scatter([], [], color='blue', marker='o', label='Realistic Predictions')
# ax.scatter([], [], color='m', marker='o', label='Unrealistic Predictions')

# Scatter plots for predicted vs actual values
ax.scatter(actual, predicted, marker='.', alpha=0.3, c='b', s=500, edgecolors='black', label='Predicted')
ax.scatter(actual, actual, marker='.', alpha=0.3, c='m', s=500, edgecolors='black', label='Actual')

# Plot y=x reference line
ax.plot([-20, 800], [-20, 800], color='black', linestyle='--')

# Set limits
ax.set_xlim([-20, 800])
ax.set_ylim([-20, 800])

# Set labels and title
ax.set_title('LSTM', fontsize=20)
ax.set_xlabel('True PCT', fontsize=20)
ax.set_ylabel('Predicted PCT', fontsize=20)

# Adjust tick label sizes
ax.tick_params(axis='both', labelsize=20)

# Add legend
ax.legend(fontsize=15, loc='upper left')

# Save and show figure
fig.savefig("pure_lstm.png")
plt.show()
# nrow = 4
# ncol = 3
# fig, ax = plt.subplots(nrow, ncol, figsize=(15,15), dpi=200)
# RD = np.random.RandomState(seed=66)
# idx_grid = RD.choice(300, (nrow, ncol), replace=False)

# idx_grid = np.array([[134, 135, 235],
#                      [233, 97, 157],
#                      [290, 239, 119],
#                      [209, 122, 152]])
# plot_idx = 1
# for i in range(nrow):
#     for j in range(ncol):
#         idx = idx_grid[i, j]
        
#         xx = [x for x in range(67) if testdata[0][1][idx][x] == 1 ]

#         yy_pred = (y_pred[idx])[testdata[0][1][idx] == 1]

#         yy_real = (y_real[idx])[testdata[0][1][idx] == 1]

#         #yy_loc = test_loc[idx][testdata[0][1][idx] == 1]

#         ax[i, j].plot(xx, yy_pred, marker='o')
#         ax[i, j].plot(xx, yy_real, marker='s')
#         #ax[i, j].plot(xx, yy_loc, marker='d')
#         ax[i, j].set_title(plot_idx)
#         plot_idx += 1
#         # ax[i, j].legend(['Predicted PCT', 'Real PCT', 'PCT Bound'])

#         ax[i, j].set_ylabel('Time')
#         ax[i, j].set_xlabel('Products')
#         ax[i, j].xaxis.set_label_coords(0.5, 0.1)

#         #for tick in ax[i, j].xaxis.get_major_ticks():
#         #    tick.label.set_fontsize(15)
#         #for tick in ax[i, j].yaxis.get_major_ticks():
#         #    tick.label.set_fontsize(15)
        
#         ax[i, j].tick_params(axis = "x", which = "both", bottom = False, top = False)
#         ax[i, j].set_xticklabels([])
#         ax[i, j].invert_xaxis()


# fig.show()
# fig.savefig('prediction_grid.png')


