import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
import xgboost as xgb
import os
import warnings
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, TimeDistributed, Flatten, Dropout, RepeatVector
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from math import sqrt


# 1. Exploration and Cleaning
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# # Read the datasets
df_weather = pd.read_csv(
    'weather_features.csv', 
    parse_dates=['dt_iso']
)
df_energy = pd.read_csv(
    'energy_dataset.csv',
    parse_dates=['time']
)

# # 1.1. Energy dataset
# print(df_energy.tail())
# Drop unusable columns
df_energy = df_energy.drop(
    ['generation fossil coal-derived gas','generation fossil oil shale', 
    'generation fossil peat', 'generation geothermal', 
    'generation hydro pumped storage aggregated', 'generation marine', 
    'generation wind offshore', 'forecast wind offshore eday ahead',
    'total load forecast', 'forecast solar day ahead',
    'forecast wind onshore day ahead',], 
    axis=1
)

# print(df_energy.describe().round(2))
# print(df_weather.describe().round(2))

# # print(df_energy.info())

# Convert time to datetime object and set it as index
df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True, infer_datetime_format=True)
df_energy = df_energy.set_index('time')


# # Find NaNs and duplicates in df_energy
# # print('There are {} missing values or NaNs in df_energy.'
# #       .format(df_energy.isnull().values.sum()))
# # temp_energy = df_energy.duplicated(keep='first').sum()
# # print('There are {} duplicate rows in df_energy based on all columns.'
# #       .format(temp_energy))


# # Find the number of NaNs in each column
# # print(df_energy.isnull().sum(axis=0))


# Define a function to plot different types of time-series
def plot_series(df=None, column=None, series=pd.Series([]), 
                label=None, ylabel=None, title=None, start=0, end=None):
    """
    Plots a certain time-series which has either been loaded in a dataframe
    and which constitutes one of its columns or it a custom pandas series 
    created by the user. The user can define either the 'df' and the 'column' 
    or the 'series' and additionally, can also define the 'label', the 
    'ylabel', the 'title', the 'start' and the 'end' of the plot.
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=16)
    if column:
        ax.plot(df[column][start:end], label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if series.any():
        ax.plot(series, label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if label:
        ax.legend(fontsize=16)
    if title:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax


# Zoom into the plot of the hourly (actual) total load
ax = plot_series(df=df_energy, column='total load actual', ylabel='Total Load (MWh)',
                 title='Total Load (First week)', end=24*7)
plt.show()


# # Display the rows with null values
# # print(df_energy[df_energy.isnull().any(axis=1)].tail())

# Fill null values using interpolation
df_energy.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)


# # Display the number of non-zero values in each column
# # print('Non-zero values in each column:\n', df_energy.astype(bool).sum(axis=0), sep='\n')


# # 1.2. Weather features dataset

# # print(df_weather.head())

# # print(df_weather.describe().round(2))

# # Print the type of each variable in df_weather
# # print(df_weather.info())

# df_energy.info()

def df_convert_dtypes(df, convert_from, convert_to):
    cols = df.select_dtypes(include=[convert_from]).columns
    for col in cols:
        df[col] = df[col].values.astype(convert_to)
    return df

# Convert columns with int64 type values to float64 type
df_weather = df_convert_dtypes(df_weather, np.int64, np.float64)

# Convert dt_iso to datetime type, rename it and set it as index
df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True, infer_datetime_format=True)
df_weather = df_weather.drop(['dt_iso'], axis=1)
df_weather = df_weather.set_index('time')


# Drop columns with qualitative weather information
df_weather = df_weather.drop(['weather_main', 'weather_id', 
                              'weather_description', 'weather_icon'], axis=1)


                                              
# # print('There are {} duplicate rows in df_weather ' \
# #       'based on all columns except "time" and "city_name".'.format(temp_weather))




# Replace outliers in 'pressure' with NaNs
df_weather.loc[df_weather.pressure > 1051, 'pressure'] = np.nan
df_weather.loc[df_weather.pressure < 931, 'pressure'] = np.nan


# Replace outliers in 'wind_speed' with NaNs
df_weather.loc[df_weather.wind_speed > 50, 'wind_speed'] = np.nan


# Fill null values using interpolation
df_weather.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

# # 1.3. Merging the two datasets

# Split the df_weather into 5 dataframes (one for each city)
df_1, df_2, df_3, df_4, df_5 = [x for _, x in df_weather.groupby('city_name')]
dfs = [df_1, df_2, df_3, df_4, df_5]

# Merge all dataframes into the final dataframe
df_final = df_energy
for df in dfs:
    city = df['city_name'].unique()
    city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
    df = df.add_suffix('_{}'.format(city_str))
    df_final = df_final.merge(df, on=['time'], how='outer')
    df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)
# print(df_final.columns)


# # 2. Visualizations and Time Series Analysis
# # 2.1. Useful visualizations and insights

cities = ['Barcelona', 'Bilbao', 'Madrid', 'Seville', 'Valencia']
for city in cities:
    df_final = df_final.drop(['rain_3h_{}'.format(city)], axis=1)



# Plot the actual electricity price at a daily/weekly scale
ax = plot_series(df_final, 'price actual', label='Hourly', ylabel='Actual Price',
                 start=1 + 24 * 500, end=1 + 24 * 522,
                 title='Actual Hourly Electricity Price for 3 Weeks')
plt.show()


# # 2.3. Autocorrelation, partial autocorrelation and cross-correlation


# Find the correlations between the electricity price and the rest of the features
correlations = df_final.corr(method='pearson')
print(correlations['price actual'].sort_values(ascending=False).to_string())

df_final = df_final.drop(['snow_3h_Barcelona', 'snow_3h_Seville','generation solar','Irradiance_Madrid','Irradiance_Valencia','Irradiance_Bilbao', 'Irradiance_Barcelona'], axis=1)



# # 3.1. Working with Features 
# # 3.1. Feature generation

# Generate 'hour', 'weekday' and 'month' features
for i in range(len(df_final)):
    position = df_final.index[i]
    hour = position.hour
    weekday = position.weekday()
    month = position.month
    df_final.loc[position, 'hour'] = position.hour
    df_final.loc[position, 'weekday'] = position.weekday()
    df_final.loc[position, 'month'] = position.month

# Generate 'business hour' feature
for i in range(len(df_final)):
    position = df_final.index[i]
    hour = position.hour
    if ((hour > 8 and hour < 14) or (hour > 16 and hour < 21)):
        df_final.loc[position, 'business hour'] = 2
    elif (hour >= 14 and hour <= 16):
        df_final.loc[position, 'business hour'] = 1
    else:
        df_final.loc[position, 'business hour'] = 0

# Generate 'weekend' feature
for i in range(len(df_final)):
    position = df_final.index[i]
    weekday = position.weekday()
    if (weekday == 6):
        df_final.loc[position, 'weekday'] = 2
    elif (weekday == 5):
        df_final.loc[position, 'weekday'] = 1
    else:
        df_final.loc[position, 'weekday'] = 0

# Generate 'temp_range' for each city
cities = ['Barcelona', 'Bilbao', 'Madrid', 'Seville', 'Valencia']
for i in range(len(df_final)):
    position = df_final.index[i]
    for city in cities:
        temp_max = df_final.loc[position, 'temp_max_{}'.format(city)]
        temp_min = df_final.loc[position, 'temp_min_{}'.format(city)]
        df_final.loc[position, 'temp_range_{}'.format(city)] = abs(temp_max - temp_min)


df_final['generation coal all'] = df_final['generation fossil hard coal'] + df_final['generation fossil brown coal/lignite']


# 3.2. Feature selection

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])
  
    return np.array(data), np.array(labels)

train_end_idx = 27048
cv_end_idx = 31056
test_end_idx = 35064

X = df_final[df_final.columns.drop('price actual')].values
y = df_final['price actual'].values
y = y.reshape(-1, 1)

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_X.fit(X[:train_end_idx])
scaler_y.fit(y[:train_end_idx])

X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)

pca = PCA()
X_pca = pca.fit(X_norm[:train_end_idx])

pca = PCA(n_components=0.80)
pca.fit(X_norm[:train_end_idx])
X_pca = pca.transform(X_norm)


# # print(X_pca.shape)

dataset_norm = np.concatenate((X_pca, y_norm), axis=1)
past_history = 24
future_target = 0

X_train, y_train = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                     0, train_end_idx, past_history, 
                                     future_target, step=1, single_step=True)

X_val, y_val = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                 train_end_idx, cv_end_idx, past_history, 
                                 future_target, step=1, single_step=True)


X_test, y_test = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                   cv_end_idx, test_end_idx, past_history, 
                                   future_target, step=1, single_step=True)
                                   
batch_size = 32
buffer_size = 1000

train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train = train.cache().shuffle(buffer_size).batch(batch_size).prefetch(1)
validation = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation = validation.batch(batch_size).prefetch(1)

# Define some common parameters

input_shape = X_train.shape[-2:]
loss = tf.keras.losses.MeanSquaredError()
metric = [tf.keras.metrics.RootMeanSquaredError()]
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
              lambda epoch: 1e-4 * 10**(epoch / 10))
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)

y_test = y_test.reshape(-1, 1)
y_test_inv = scaler_y.inverse_transform(y_test)

# # 4. Electricity Price Forecasting

# 4.2. XGBoost

X_train_xgb = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_val_xgb = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
X_test_xgb = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

param = {'eta': 0.03, 'max_depth': 6, 'random_state': 1,
         'subsample': 1.0, 'colsample_bytree': 0.95, 
         'alpha': 0.1, 'lambda': 0.15, 'gamma': 0.1,
         'objective': 'reg:squarederror', 'eval_metric': 'rmse', 
         'verbosity': 0, 'min_child_weight': 0.1, 'n_jobs': -1}

dtrain = xgb.DMatrix(X_train_xgb, y_train)
dval = xgb.DMatrix(X_val_xgb, y_val)
dtest = xgb.DMatrix(X_test_xgb, y_test)
eval_list = [(dtrain, 'train'), (dval, 'eval')]

xgb_model = xgb.train(param, dtrain, 180, eval_list, early_stopping_rounds=3)

forecast = xgb_model.predict(dtest)

xgb_forecast = forecast.reshape(-1, 1)

xgb_forecast_inv = scaler_y.inverse_transform(xgb_forecast)


# fig, ax1 = plt.subplots()
# ax2 =ax1.twinx()
# hour = [ x for x in range(24)]
# hour1 = hour + [x+24 for x in hour]
# ax1.plot(hour1, total_energy, label='Energy')
# ax2.plot(hour1, price, label='price')
# ax2.plot(hour1, fore_price,  label='fore price')
# # ax1.plot(hour1, total_energy[24:], 'g-')
# # ax2.plot(hour1, fore_price[24:], 'b-')
# ax1.set_xlabel('Hour')
# ax1.set_ylabel('Energy (KWh)')
# ax2.set_ylabel('Cost (Euro)')
# plt.grid(True)
# plt.show()

rmse_xgb = sqrt(mean_squared_error(y_test_inv, xgb_forecast_inv))
print('XGBoost RMSE of price forecast: {}'
      .format(round(rmse_xgb, 3)))

print('yyyyyyyyyyy', len(y_test_inv))
print('xxxxxxxxxx', len(xgb_forecast_inv))

# # # 4.3. LSTM

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

multivariate_lstm = tf.keras.models.Sequential([
    LSTM(100, input_shape=input_shape, 
         return_sequences=True),
    Flatten(),
    Dense(200, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_lstm.keras', monitor=('val_loss'), save_best_only=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=6e-3, amsgrad=True)

multivariate_lstm.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metric)

history = multivariate_lstm.fit(train, epochs=120,
                                validation_data=validation,
                                callbacks=[early_stopping, 
                                           model_checkpoint])

# # # plot_model_rmse_and_loss(history)

multivariate_lstm = tf.keras.models.load_model('multivariate_lstm.keras')

forecast = multivariate_lstm.predict(X_test)
lstm_forecast = scaler_y.inverse_transform(forecast)

rmse_lstm = sqrt(mean_squared_error(y_test_inv,
                                    lstm_forecast))
print('LSTM RMSE of price forecast: {}'
      .format(round(rmse_lstm, 3)))

# # # 4.4. Stacked LSTM

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

multivariate_stacked_lstm = tf.keras.models.Sequential([
    LSTM(250, input_shape=input_shape, 
         return_sequences=True),
    LSTM(150, return_sequences=True),
    Flatten(),
    Dense(150, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_stacked_lstm.keras', save_best_only=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3, amsgrad=True)

multivariate_stacked_lstm.compile(loss=loss,
                                  optimizer=optimizer,
                                  metrics=metric)

history = multivariate_stacked_lstm.fit(train, epochs=120,
                                validation_data=validation,
                                callbacks=[early_stopping, 
                                           model_checkpoint])

# # # plot_model_rmse_and_loss(history)

multivariate_stacked_lstm = tf.keras.models.load_model('multivariate_stacked_lstm.keras')

forecast = multivariate_stacked_lstm.predict(X_test)
multivariate_stacked_lstm_forecast = scaler_y.inverse_transform(forecast)

rmse_mult_stacked_lstm = sqrt(mean_squared_error(y_test_inv, 
                                                 multivariate_stacked_lstm_forecast))
print('LSTM RMSE of price forecast: {}'
      .format(round(rmse_mult_stacked_lstm, 3)))

# # # 4.5. CNN

# tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

multivariate_cnn = tf.keras.models.Sequential([
    Conv1D(filters=48, kernel_size=2,
           strides=1, padding='causal',
           activation='relu', 
           input_shape=input_shape),
    Flatten(),
    Dense(48, activation='relu'),
    Dense(1)
])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_cnn.keras', save_best_only=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=6e-3, amsgrad=True)

multivariate_cnn.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metric)

history = multivariate_cnn.fit(train, epochs=120,
                               validation_data=validation,
                               callbacks=[early_stopping, 
                                          model_checkpoint])

# # # plot_model_rmse_and_loss(history)

multivariate_cnn = tf.keras.models.load_model('multivariate_cnn.keras')

forecast = multivariate_cnn.predict(X_test)
multivariate_cnn_forecast = scaler_y.inverse_transform(forecast)

rmse_mult_cnn = sqrt(mean_squared_error(y_test_inv,
                                        multivariate_cnn_forecast))
print('CNN RMSE of price forecast: {}'
      .format(round(rmse_mult_cnn, 3)))

# # # 4.6. CNN-LSTM

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

multivariate_cnn_lstm = tf.keras.models.Sequential([
    Conv1D(filters=100, kernel_size=2,
           strides=1, padding='causal',
           activation='relu', 
           input_shape=input_shape),
    LSTM(100, return_sequences=True),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_cnn_lstm.keras', save_best_only=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=4e-3, amsgrad=True)

multivariate_cnn_lstm.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metric)

history = multivariate_cnn_lstm.fit(train, epochs=120,
                                    validation_data=validation,
                                    callbacks=[early_stopping, 
                                               model_checkpoint])

# # # plot_model_rmse_and_loss(history)

multivariate_cnn_lstm = tf.keras.models.load_model('multivariate_cnn_lstm.keras')

forecast = multivariate_cnn_lstm.predict(X_test)
multivariate_cnn_lstm_forecast = scaler_y.inverse_transform(forecast)

rmse_mult_cnn_lstm = sqrt(mean_squared_error(y_test_inv, 
                                             multivariate_cnn_lstm_forecast))
print('CNN-LSTM RMSE of price forecast: {}'
      .format(round(rmse_mult_cnn_lstm, 3)))

# # # 4.7. Time Distributed MLP

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

multivariate_mlp = tf.keras.models.Sequential([
    TimeDistributed(Dense(200, activation='relu'),
                    input_shape=input_shape),
    TimeDistributed(Dense(150, activation='relu')),
    TimeDistributed(Dense(100, activation='relu')),
    TimeDistributed(Dense(50, activation='relu')),
    Flatten(),
    Dense(150, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_mlp.keras', save_best_only=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3, amsgrad=True)

multivariate_mlp.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metric)

history = multivariate_mlp.fit(train, epochs=120,
                               validation_data=validation,
                               callbacks=[early_stopping, 
                                          model_checkpoint])

# # # plot_model_rmse_and_loss(history)

multivariate_mlp = tf.keras.models.load_model('multivariate_mlp.keras')

forecast = multivariate_mlp.predict(X_test)
multivariate_mlp_forecast = scaler_y.inverse_transform(forecast)

rmse_mult_mlp = sqrt(mean_squared_error(y_test_inv,
                                        multivariate_mlp_forecast))
print('MLP RMSE of price forecast: {}'
      .format(round(rmse_mult_mlp, 3)))



# AC optimization using strategy
# estimated duty cycle in minute
energy = {
    '16': 50/60 * 2110/1000,
    '24': 35/60 * 2110/1000,
}

cost = []
total_energy = []
hour = []

fore_price = xgb_forecast_inv[-3959:-23 ,0]
price = y_test_inv[-3960:-24, 0]
history = y_test_inv[:-24, 0]

def optimization(price, fore_price):
    if fore_price < price  :
        # ac off work on 24
        est_energy = energy['16']
    else : 
         # ac off work on 16
        est_energy = energy['24']
    total_energy.append(est_energy)
    cost.append((fore_price/1000) * est_energy)
x = len(y_test_inv)
for i in range(x):
    try:
        optimization(price[i], fore_price[i])
    except IndexError:
        continue

print('cost', len(cost) ,sum(cost))
print('total energy', len(total_energy), sum(total_energy))
print("\n")


# operating on 16 only
# estimated duty cycle in minute
energy = {
    '16': 50/60 * 2110/1000,
    '24': 35/60 * 2110/1000,
}

fore_price = xgb_forecast_inv[-3959:-23 ,0]
price = y_test_inv[-3960:-24, 0]
history = y_test_inv[:-24, 0]

def optimization(price, fore_price):
    est_energy = energy['16']
    total_energy.append(est_energy)
    cost.append((fore_price/1000) * est_energy)
x = len(y_test_inv)
for i in range(x):
    try:
        optimization(price[i], fore_price[i])
    except IndexError:
        continue

print('cost', len(cost) ,sum(cost))
print('total energy', len(total_energy), sum(total_energy))
print("\n")


# operating on 16 only
# estimated duty cycle in minute

cost = []
total_energy = []
hour = []

fore_price = xgb_forecast_inv[-3959:-23 ,0]
price = y_test_inv[-3960:-24, 0]
history = y_test_inv[:-24, 0]

def optimization(price, fore_price):
    est_energy = energy['24']
    total_energy.append(est_energy)
    cost.append((fore_price/1000) * est_energy)
x = len(y_test_inv)
for i in range(x):
    try:
        optimization(price[i], fore_price[i])
    except IndexError:
        continue

print('cost', len(cost) ,sum(cost))
print('total energy', len(total_energy), sum(total_energy))
print("\n")