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
# %%
#test for opencv
import cv2
import numpy as np

# %%
image = cv2.imread('opencv test.jpg')
# %%
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
(_, cnts) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# %%
img = cv2.imread('opencv2.jpg')
# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

# %%
cv2.imshow('windows_name',img)
cv2.waitKey (0) 
cv2.destroyAllWindows()
cv2.waitKey(1)


# %%
