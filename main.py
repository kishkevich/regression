


import warnings

import inline as inline
import matplotlib

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data=pd.read_csv('train.csv')

print (data.head())
print (data.describe())
print (data.isnull().sum())
print(data.info())

def impute_NA_with_avg(data, strategy='mean', NA_col=[]):
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            if strategy == 'mean':
                data_copy[i + '_impute_mean'] = data_copy[i].fillna(data[i].mean())
            else:
                warnings.warn("Нет пропущенных значений" % i)
    return data_copy
new_data = impute_NA_with_avg(data=data, strategy='mean', NA_col=['price'])

print(new_data.mean())

data.price.hist(bins=30)

data.drop('price', axis=1).hist(figsize=(30, 20));

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(70, 40))
sns.heatmap(data.corr(), annot=True);

import seaborn as sns

sns.pairplot(data)

data_reduction = new_data.drop( ['id', 'month', 'surge_multiplier', 'latitude', 'longitude', 'precipIntensity', 'uvIndex', 'visibility',
     'visibility.1', 'precipIntensityMax'], axis=1)
print (data_reduction.head(5))

data_object = data_reduction.select_dtypes(include='object')
print (data_object.head(5))

#удаление данных типа object
data_without_object = new_data.drop(data.select_dtypes(include=['object']), axis=1)
print (data_without_object.head(5))
print (data_without_object.info())

#деление на тренировочные и тестовые данные
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(data_without_object, test_size=0.3)

#скалирование
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_train[['month', 'longitude', 'latitude', 'temperatureMin', 'moonPhase']])
df_train_scale = scaler.transform(df_train[['month', 'longitude', 'latitude', 'temperatureMin', 'moonPhase']])
df_test_scale = scaler.transform(df_test[['month', 'longitude', 'latitude', 'temperatureMin', 'moonPhase']])
df_train[['month', 'longitude', 'latitude', 'temperatureMin', 'moonPhase']] = df_train_scale
df_test[['month', 'longitude', 'latitude', 'temperatureMin', 'moonPhase']] = df_test_scale
print (df_train.head())


y_train = df_train.price
y_test = df_test.price

X_train = df_train.drop(['price'], axis=1)
X_test = df_test.drop(['price'], axis=1)

y_mean = np.mean(y_train)
y_pred_naive = np.ones(len(y_test)) * y_mean
y_pred_naive[:5]


from sklearn import metrics
def print_metrics(y_test, y_pred):
    print('MAE:', metrics.mean_absolute_error(np.exp(y_test), np.exp(y_pred)))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(np.exp(y_test), np.exp(y_pred))))
    print('R2:', metrics.r2_score(y_test, y_pred))
    print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))
    pass
print_metrics(y_test, y_pred_naive)



from sklearn.linear_model import LinearRegression

model_regression = LinearRegression()
model_regression.fit(X_train, y_train)
y_pred_regr = model_regression.predict(X_test)
print_metrics(y_test, y_pred_regr)
featureImportance = pd.DataFrame({"feature": df.drop('price', axis=1).columns,
                                  "importance": model_regression.coef_})
featureImportance.set_index('feature', inplace=True)
featureImportance.sort_values(["importance"], ascending=False, inplace=True)
featureImportance["importance"].plot('bar', figsize=(10, 6));
