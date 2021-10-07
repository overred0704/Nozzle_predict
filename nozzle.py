#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
data = pd.read_excel('智慧化燃燒器V5.xlsx')
# %%
data.columns = ['Afr', 'power','caliber','up_pa','down_pa','diff_pa','temp_k','temp_c','smoke_temp','fp']
corr = data.corr()

# %%
sns.heatmap(corr, annot = True )
plt.show()
# %%
#select data
df = data[['caliber', 'up_pa','down_pa','smoke_temp','fp']]
dfvalue = df.values
dfvalue[:, 1:]
# %%
#curve fitting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# %%
model = LinearRegression()
y = df.iloc[:, :1]
x = df.iloc[:, 1:]
# %%
model.fit(x,y)
# %%
y_pred = model.predict(x)
# %%
r2_score(y, y_pred)
# %%
model.intercept_
# %%
#try xgboost
from  xgboost import XGBRegressor
# %%
xg = XGBRegressor()
# %%
xg.fit(x,y)
# %%
xg.score(x,y)


#%%
#1006 test for two group
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
# %%
data = pd.read_excel('智慧化燃燒器V5.xlsx')
data.columns = ['Afr', 'power','caliber','up_pa','down_pa','diff_pa','temp_k','temp_c','smoke_temp','fp']
# %%
df_1 = data[['caliber', 'up_pa','down_pa','smoke_temp','fp']]
df_2 = data[['caliber', 'smoke_temp','fp']]

#%%
df_y = df_1['caliber']*100

#%%
df_1_x = df_1[['up_pa','down_pa','smoke_temp','fp']]
df_2_x = data[['smoke_temp','fp']]
# %%
model_1 = sm.OLS(df_y, df_1_x)
results_1 = model_1.fit()
print(results_1.summary())
# %%
model_2 = sm.OLS(df_y, df_2_x)
results_2 = model_2.fit()
print(results_2.summary())
# %%
#sklearn linear model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# %%
model_3 = LinearRegression()
model_3.fit(df_1_x,df_y)
y_pred_1 = model_3.predict(df_1_x)

coef_raw = model_3.coef_
coef = [round(num, 2) for num in coef_raw]
print(f'擴張口徑 = 上游壓力*{coef[0]} + 下游壓力*{coef[1]} + 煙氣溫度*{coef[2]} + 爐壓*{coef[3]}')
print(f'r2 = {r2_score(df_y, y_pred_1)}')
# %%
model_4 = LinearRegression()
model_4.fit(df_2_x,df_y)
y_pred_2 = model_4.predict(df_2_x)
coef_raw = model_4.coef_
coef = [round(num, 2) for num in coef_raw]
print(f'擴張口徑 = 煙氣溫度*{coef[0]} + 爐壓*{coef[1]}')
print(f'r2 = {r2_score(df_y, y_pred_2)}')

#%%
data = pd.read_excel('智慧化燃燒器V5.xlsx')
data.columns = ['Afr', 'power','caliber','up_pa','down_pa','diff_pa','temp_k','temp_c','smoke_temp','fp']
df_1 = data['caliber']
df_2 = data[['up_pa','down_pa','smoke_temp','fp']]
df_1 = df_1*100


# %%
for i in df_1.columns:
    df_1[i] = (df_1[i] - df_1[i].mean())/df_1[i].std()
#%%
df_1