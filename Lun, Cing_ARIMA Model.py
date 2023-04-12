#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


# In[3]:


import sklearn
from sklearn.metrics import mean_squared_error


# # Load in Data

# ## The data was filter from Wells Fargo personal credit card bank statement.

# ## This data is filter only for the year of 2022 through the csv file in Excel.

# In[4]:


CreditCard2022 = pd.read_csv('C:/Users/Cing San Lun/Desktop/Bethel/DSO110 - Final Group Project/Lun, Cing_DSO110-final-Project/Data-Clean/Credit Card Year End 2022.csv')
CreditCard2022.head()


# In[5]:


CreditCard2022.tail()


# # Check the shape of the data

# In[6]:


CreditCard2022.shape


# # Remove 'Payment Method' Column

# In[7]:


CreditCard2022.drop('Payment Method', axis=1, inplace=True)


# In[8]:


CreditCard2022.head()


# # Split the Date column into Month, Day, and Year column

# In[9]:


CreditCard2022[['Month', 'Day', 'Year']] = CreditCard2022['Date'].str.split("/", expand = True)


# ## Convert the 'Year' column to integer format

# In[10]:


CreditCard2022 = CreditCard2022.astype({'Year':'int'})


# In[11]:


CreditCard2022.head()


# # Convert the 'Amount' column to currency format

# ## I can now do math operation specifically on 'Amount column'

# In[12]:


CreditCard2022['Amount'] = CreditCard2022['Amount'].replace('[^\d.]', '', regex=True).astype(float)


# In[13]:


CreditCard2022.info()


# In[14]:


CreditCard2022_ap = CreditCard2022[['Master Category', 'Year', 'Month', 'Amount']]


# # Check the overall info

# In[15]:


CreditCard2022_ap.info()


# In[16]:


CreditCard2022_ap.describe()


# # Check Missing values

# ## There is no missing values in the data. We do not need to treat them.

# In[17]:


CreditCard2022_ap.isnull().sum()


# # Outliers

# In[18]:


CreditCard2022_ap.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# In[19]:


plt.figure(figsize=(12,6))
sns.boxplot(CreditCard2022['Amount'])
plt.show()


# ## There are outliers in the data. They are greater than upper fence.

# In[20]:


plt.figure(figsize=(12,6))
plt.hist(CreditCard2022['Amount'])
plt.show()


# ## Right Skewed Histogram

# In[21]:


CreditCard2022_ap.info()


# ## Transform date into a format acceptable by model

# In[22]:


CreditCard2022_ap['Date'] = pd.to_datetime(CreditCard2022['Date'])


# In[23]:


CreditCard2022_ap.head()


# # Group the data by Date to aggregate Master Category

# In[24]:


CreditCard20221 = CreditCard2022_ap.groupby(['Date'])['Amount'].median().reset_index()
CreditCard20221.head()


# # Convert to Data Frame

# In[25]:


CreditCard20221 = pd.DataFrame(CreditCard20221)


# In[26]:


CreditCard20221 = CreditCard20221.set_index(['Date'])


# In[27]:


CreditCard20221.head()


# In[28]:


CreditCard20221.index.dtype


# In[29]:


plt.figure(figsize=(18,4))
plt.plot(CreditCard20221, label='Index')
plt.legend(loc='best')
plt.title('Consumer Price Index\n', fontdict={'fontsize': 16, 'fontweight': 5, 'color': 'Orange'})
plt.xticks(rotation = 90, fontweight ="bold")
plt.show()


# ## There are fluctuations in this plot.

# # Timeseries decompose into different parts
# ## Additive seasonal decomposition

# In[30]:


from pylab import rcParams
from scipy.fftpack import fftfreq
import statsmodels.api as sm
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(CreditCard20221['Amount'], model='additive', period=12)
fig = decomposition.plot()
plt.show()


# # Multiplicative seasonal decomposition

# In[31]:


decomposition = sm.tsa.seasonal_decompose(CreditCard20221['Amount'], model='multiplicative', period=12)
fig = decomposition.plot()
plt.show()


# ## The best model is Multiplicative seasonal decomposition

# # Split the data into train and then test the sets

# In[32]:


train_len = 122
train = CreditCard20221[0 : train_len]
test = CreditCard20221[train_len : ]


# In[33]:


train.head()


# In[34]:


test


# ## Build Time Series Forecast models, compare Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) values for the models.

# # Check for stationary within the time series

# ### Average/Mean/Expected value of the time series is constant throughout different periods.
# ### Standard Deviation/Variance of the time series is consant throughout different periods.
# ### There is no seasonality in the data.

# ## Perform tansformation and differencing

# In[35]:


plt.figure(figsize=(12, 4))
plt.plot(CreditCard20221['Amount'], label='Amount')
plt.legend(loc='best')
plt.title('Distribution of CPI')
plt.show()


# # Augmented Dickey-Fuller (ADF) test

# In[36]:


from statsmodels.tsa.stattools import adfuller


# In[37]:


adf_test = adfuller(CreditCard20221['Amount'])
print(adf_test)
print('ADF Statistic %f' % adf_test[0])
print('Critical Values @ 0.05 %.2f' % adf_test[4]['5%'])
print('p-value: %f' %adf_test[1])


# ## p-value is less than 0.05. This means that the series is stationary.

# In[38]:


from statsmodels.tsa.stattools import kpss


# In[39]:


kpss_test = kpss(CreditCard20221['Amount'])
print('KPSS Statistic %f' % kpss_test[0])
print('Critical Values @ 0.05 %.2f' % kpss_test[3]['5%'])
print('p-value: %f' %kpss_test[1])


# ## p-value is greater than 0.05. This means that the series is stationary.

# In[40]:


from scipy.stats import boxcox 

data_boxcox = pd.Series(boxcox(CreditCard20221['Amount'], lmbda=0), index = CreditCard20221.index)
plt.figure(figsize = (12,4))
plt.plot(data_boxcox, label='After Box Cox Transformation')
plt.legend(loc='best')
plt.title('After Box Cox Transform\n', fontdict={'fontsize':16, 'fontweight':5, 'color':'Orange'})
plt.show()


# In[41]:


data_boxcox_diff = pd.Series(boxcox(CreditCard20221['Amount'], lmbda=0), index = CreditCard20221.index)
plt.figure(figsize = (15,6))
plt.plot(data_boxcox_diff, label='After Box Cox Transformation and Differencing')
plt.legend(loc='best')
plt.title('After Box Cox Transform and Differencing\n', fontdict={'fontsize':16, 'fontweight':5, 'color':'Orange'})
plt.show()


# # Train-Test Split

# In[42]:


train_data_boxcox = data_boxcox[:train_len]
test_data_boxcox = data_boxcox[train_len:]
train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
test_data_boxcox_diff = data_boxcox_diff[train_len-1:]


# # Build various AR models to forecast the Amount

# In[43]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data_boxcox_diff)


# In[44]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data_boxcox_diff)


# ## Helps us define the parameter of the model

# # ARIMA model

# In[45]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data_boxcox_diff, order=(2, 1, 9))
model_fit = model.fit()

import warnings
warnings.filterwarnings('ignore')


# ### Find the original time series

# In[50]:


data_ar = data_boxcox_diff.copy()
data_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
data_ar['ar_forecast_boxcox'] = data_ar['ar_forecast_boxcox_diff'].cumsum()
data_ar['ar_forecast_boxcox'] = data_ar['ar_forecast_boxcox'].add(data_boxcox[0])
data_ar['ar_forecast'] = np.exp(data_ar['ar_forecast_boxcox'])


# ### Plotting, Train, and forecast

# In[51]:


plt.figure(figsize=(12, 4))
plt.plot(train['Amount'], label='Train')
plt.plot(test['Amount'], label='Test')
plt.plot(data_ar['ar_forecast'][test.index.min():], label='Auto Regression Forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
plt.show()


# ### Calcualte RMSE and MAPE

# In[52]:


RMSE = np.sqrt(mean_squared_error(test['Amount'], data_ar['ar_forecast'][test.index.min():])).round(2)
MAPE = np.round(np.mean(np.abs(test['Amount']-data_ar['ar_forecast'][test.index.min():])/test['Amount'])*100,2)
print(MAPE)


# In[55]:


Predict_Future = model_fit.predict(start = len(CreditCard20221), end=(len(CreditCard20221)) + 6)
Predict_Future = Predict_Future.cumsum()
Predict_Future = Predict_Future.add(data_boxcox[0])
Predict_Future = np.exp(Predict_Future)


# In[56]:


Predict_Future.plot()
plt.title('Next Six Months Prediction\n', fontdict={'fontsize': 16, 'fontweight': 5, 'color': 'Orange'})
plt.show()

