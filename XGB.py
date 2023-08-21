#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:56:57 2023

@author: leminhtri
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

##########################     Tải dữ liệu    #################################

df = pd.read_excel('/Users/leminhtri/df_join_separate_dateyear.xlsx', parse_dates=True)
df = pd.DataFrame({"year": df["year"],"closeADC": df["closeADC"]})
df

#Biến chính close price và các chỉ số 
#EMA_9, SMA_5, SMA_10, SMA_15, SMA_30, RSI, MACD, MACD_signal
#là các thuộc tính đầu vào cho XGB. 
#SMA (Simple Moving Average) đại diện cho giá trung bình của một khoảng thời gian cụ thể
#EMA (Exponential Moving Average) là một biến thể của SMA nhưng trọng số được áp dụng cho các giá trị gần đây hơn
#RSI (Relative Strength Index) đo lường sức mạnh của một xu hướng  
#MACD (Moving Average Convergence Divergence) được sử dụng để xác định sự khác biệt giữa hai giá trị trung bình di động khác nhau.

#MA
#Loại bỏ các dòng có giá trị null trong cột 'closeADC'
#df = df.dropna(subset=['closeADC']) 
##dòng trên để hạn chế và loại bỏ gía trị null trong tập dữ liệu, null sẽ làm ảnh hưởng đến quá trình tính toán 

smoothing_value = 0.1
##thêm giá trị smoothing nhỏ vào các biến tính toán để tránh giá trị 0 trong dữ liệu tính toán và tăng tính ổn định của các chỉ số 
df['EMA_9'] = df['closeADC'].ewm(9).mean().shift().fillna(method='ffill') + smoothing_value
df['SMA_5'] = df['closeADC'].rolling(5).mean().shift().fillna(method='ffill') + smoothing_value
df['SMA_10'] = df['closeADC'].rolling(10).mean().shift().fillna(method='ffill') + smoothing_value
df['SMA_15'] = df['closeADC'].rolling(15).mean().shift().fillna(method='ffill') + smoothing_value
df['SMA_30'] = df['closeADC'].rolling(30).mean().shift().fillna(method='ffill') + smoothing_value

plt.figure(figsize=(12,7))
plt.plot(df['EMA_9'], label='EMA 9')
plt.plot(df['SMA_5'], label='SMA 5')
plt.plot(df['SMA_10'], label='SMA 10')
plt.plot(df['SMA_15'], label= 'SMA 15')
plt.plot(df['SMA_30'], label= 'SMA 30')
plt.plot(df['closeADC'], label = 'Close')
plt.xticks(np.arange(0,1251, 200), df['year'][0:1251:200])
plt.legend()

#RSI
def relative_strength_idx(df, n=14):
    close = df['closeADC']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

df['RSI'] = relative_strength_idx(df).fillna(method='ffill') + smoothing_value

plt.figure(figsize=(12,7))
plt.plot(df['RSI'], label='RSI')
plt.xticks(np.arange(0,1251, 200), df['year'][0:1251:200])
plt.legend()

#MACD
EMA_12 = pd.Series(df['closeADC'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['closeADC'].ewm(span=26, min_periods=26).mean())
df['MACD'] = pd.Series(EMA_12-EMA_26).fillna(method='ffill') + smoothing_value
df['MACD_signal'] = pd.Series(df['MACD'].ewm(span=9, min_periods=9).mean()).fillna(method='ffill') + smoothing_value

plt.figure(figsize=(12,7))
plt.plot(EMA_12, label='EMA_12')
plt.plot(EMA_26, label='EMA_26')
plt.plot(df['MACD'], label='MACD')
plt.plot(df['MACD_signal'], label= 'Signal Line')
plt.plot(df['closeADC'], label = 'Close')
plt.xticks(np.arange(0,1251, 200), df['year'][0:1251:200])
plt.legend()

###########################    Xử lý dữ liệu       ############################

#Đoạn code này dịch chuyển dữ liệu trong cột "closeADC" lên một hàng
#Tức là dữ liệu tại hàng thứ i sẽ được chuyển sang hàng thứ i+1
#Tóm lại mục đích là dự đoán giá cổ phiếu vào ngày tiếp theo (dựa trên giá trị của ngày hiện tại)
#Tuy nhiên nó sẽ làm mất giá trị của hàng cuối cùng trong cột "closeADC" và thay vào đó là NaN  
#Vì vậy cần xác định liệu việc mất giá trị của hàng cuối cùng trong cột "closeADC" có ảnh hưởng 
#đến việc phân tích hay dự báo dữ liệu hay không
df['closeADC'] = df['closeADC'].shift(-1)

#Tuy nhiên, mình quyết định không loại bỏ giá trị Nah mà sẽ dùng các kĩ thuật để hoàn thiện bộ dữ liệu

#Thêm vào các giá trị bị thiếu (sử dụng giá trị trung bình của các cột để thay thế các giá trị bị thiếu)
df.fillna(df.mean(), inplace=True)

#Loại bỏ outliers (sử dụng phương pháp IQR để xác định và loại bỏ outliers)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

#Dùng PCA để xử lý đặc trưng phức tạp
#pca = PCA(n_components=2)
#complex_features = pca.fit_transform(df[['SMA_30', 'RSI']])

#Loại bỏ những dòng có giá trị  null (chỉ áp dụng khi không sử dụng cách thêm vào giá trị thiếu và loại bỏ outliers ở trên)
#df = df.iloc[33:] # Bởi vì MACD Signal không thể tính toán đến ngày thứ 32
#df = df[:-1]      # Bởi  shifting close price
#df.index = range(len(df))

######################  Chia tập test và train     ############################

#training(70%), validation (15%) và test (15%)
test_size  = 0.15
valid_size = 0.15

test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()

plt.figure(figsize=(12,7))
plt.plot(train_df['closeADC'], label='Train')
plt.plot(valid_df['closeADC'], label='Validation')
plt.plot(test_df['closeADC'], label='Test')
plt.xticks(np.arange(0,966, 150), df['year'][0:966:150])
plt.legend()

#Loại bỏ dòng k cần thiết (year)
drop_cols = ['year']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)

#y là close_ADC là biến chính
#x gồm các giá trị EMA_9, SMA_5, SMA_10, SMA_15, SMA_30, RSI, MACD, MACD_signal 
#ta đã có đủ biến quan trọng để dự đoán giá cổ phiếu như EMA, SMA, RSI, MACD

#Split into features and labels
y_train = train_df['closeADC'].copy()
X_train = train_df.drop(['closeADC'], 1)

y_valid = valid_df['closeADC'].copy()
X_valid = valid_df.drop(['closeADC'], 1)

y_test  = test_df['closeADC'].copy()
X_test  = test_df.drop(['closeADC'], 1)

X_train.info()

#y_train: Dataframe chứa giá trị closeADC của tập train
#X_train: Dataframe chứa các features của tập train (EMA, MACD)
#y_valid: Series chứa giá trị closeADC của tập valid
#X_valid: DataFrame chứa các features của tập valid
#y_test: Series chứa giá trị closeACDC của tập test
#X_test: DataFrame chứa các features của tập test

#Tập train nếu không loại bỏ outliers
#0   EMA_9        853 non-null    float64
#1   SMA_5        853 non-null    float64
#2   SMA_10       853 non-null    float64
#3   SMA_15       853 non-null    float64
#4   SMA_30       853 non-null    float64
#5   RSI          853 non-null    float64
#6   MACD         853 non-null    float64
#7   MACD_signal  853 non-null    float64

#Tập Train nếu loại bỏ outliers
#0   EMA_9        674 non-null    float64
#1   SMA_5        674 non-null    float64
#2   SMA_10       674 non-null    float64
#3   SMA_15       674 non-null    float64
#4   SMA_30       674 non-null    float64
#5   RSI          674 non-null    float64
#6   MACD         674 non-null    float64
#7   MACD_signal  674 non-null    float64
#dtypes: float64(8)

####    Thử nghiệm nhiều tham số khác nhau để tìm mô hình nào oke nhất    #####

#Mô hình 1
parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(objective='reg:squarederror')
clf = GridSearchCV(model, parameters)

clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

#Mô hình 2
parameters = {
    'learning_rate': uniform(0.001, 0.1),
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 20),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 1.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 0.05),
    'reg_lambda': uniform(0.1, 49.9),
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(objective='reg:squarederror')
clf = RandomizedSearchCV(estimator=model, 
                        param_distributions=parameters, 
                        n_iter=100, 
                        cv=5, 
                        random_state=42,
                        n_jobs=-1)
clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

#Mô hình 3
parameters = {
    "learning_rate": [0.1, 0.2, 0.3],
   "max_depth": [3, 4, 5],
   "min_child_weight": [1, 3, 5],
   "subsample": [0.5, 0.7],
   "colsample_bytree": [0.5, 0.7],
   "gamma": [0, 0.1, 0.2],
   "alpha": [0, 0.1, 0.2],
   "lambda": [0, 0.1, 0.2],
   "n_estimators": [50, 100, 200]
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(objective='reg:squarederror')
clf = GridSearchCV(model, parameters, cv=5, n_jobs=-1)

clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

#Kết quả 1
#Best validation score = 0.1679939229254836

#Kết quả 2
#Best validation score = 0.3790562998877088

#Kết quả 3
#Best validation score = 0.4125857021887013

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
# XGBRegressor(base_score=None, booster=None, callbacks=None,
            # colsample_bylevel=None, colsample_bynode=None,
            # colsample_bytree=None, early_stopping_rounds=None,
            # enable_categorical=False, eval_metric=None, feature_types=None,
            # gamma=0.01, gpu_id=None, grow_policy=None, importance_type=None,
            # interaction_constraints=None, learning_rate=0.05, max_bin=None,
            # max_cat_threshold=None, max_cat_to_onehot=None,
            # max_delta_step=None, max_depth=8, max_leaves=None,
            # min_child_weight=None, missing=nan, monotone_constraints=None,
            # n_estimators=100, n_jobs=None, num_parallel_tree=None,
            # predictor=None, random_state=42, ...)

plot_importance(model)

#Kết quả dự đoán so với giá trị thực
y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')

#Kết quả 1
#y_true = [13.6 14.7 14.7 14.7 13.3]
#y_pred = [14.550083 14.569026 14.71867  14.769511 14.809822]

#Kết qủa 2
#y_true = [13.6 14.7 14.7 14.7 13.3]
#y_pred = [14.539252 14.656349 14.77622  14.858872 14.873167]

#Kết quả 3
#y_true = [13.6 14.7 14.7 14.7 13.3]
#y_pred = [14.577321 14.588515 14.793188 15.082795 15.101603]

#Sai số dự đoán
y_test = np.nan_to_num(y_test)
y_pred = np.nan_to_num(y_pred)
print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

#nếu không lấp đầy giá trị null và loại bỏ các outliers, sai số ban đầu lên 
#tới tận 51, 60, kết quả mô hình quá thấp

#Kết quả sai số 1
#mean_squared_error = 1.1574657673599908

#Kết qủa sai số 2
#mean_squared_error = 1.0739755140524165

#Kết qủa sai số 3
#mean_squared_error = 1.1546627646976086

#In ra bảng giá dự đoán
predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['closeADC'] = y_pred


##########################      Trực quan hoá            ######################

#Cách 1: vẽ riêng lẻ
plt.figure(figsize=(12,7))
plt.plot(df['closeADC'], color='LightSkyBlue', label='Truth')
plt.plot(predicted_prices['closeADC'], color='MediumPurple', label='Prediction')
plt.xticks(np.arange(0,967, 150), df['year'][0:967:150])
plt.legend()

plt.figure(figsize=(12,7))
plt.plot(y_test, color='LightSkyBlue')
plt.plot(y_pred, color='MediumPurple')
plt.xticks(np.arange(0,170, 50), predicted_prices['year'][0:170:50])


#Cách 2: vẽ kết hợp 2 chart trong 1 bảng
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
# First chart
axs[0].plot(df['closeADC'], color='LightSkyBlue', label='Truth')
axs[0].plot(predicted_prices['closeADC'], color='MediumPurple', label='Prediction')
axs[0].set_xticks(np.arange(0, 967, 150))
axs[0].set_xticklabels(df['year'][0:967:150])
axs[0].legend()
# Second chart
axs[1].plot(y_test, color='LightSkyBlue')
axs[1].plot(y_pred, color='MediumPurple')
axs[1].set_xticks(np.arange(0, 170, 50))
axs[1].set_xticklabels(predicted_prices['year'][0:170:50])
# Set chart title and labels
fig.suptitle('Comparison of Truth and Prediction', fontsize=16)
axs[0].set_ylabel('Close ADC', fontsize=12)
axs[1].set_ylabel('Close ADC', fontsize=12)
axs[1].set_xlabel('Year', fontsize=12)

plt.show()