#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , r2_score


# In[24]:


data = pd.read_excel("NIFTY.xlsx")
data


# In[25]:


data['Date'] = data['date'].apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d").date())
data


# In[26]:


data['Date'].describe()


# In[27]:


training_set = data['open'].values.reshape(-1,1)
training_set


# training_set.shape

# In[30]:


scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(training_set)
scaled_data.shape


# In[31]:


X_axis = []
Y_axis = []
for i in range(60, len(scaled_data)):
    X_axis.append(scaled_data[i-60:i, 0])
    Y_axis.append(scaled_data[i,0])
X_axis = np.array(X_axis)
Y_axis = np.array(Y_axis)


# In[35]:


np.reshape(X_axis, (X_axis.shape[0], X_axis.shape[1], 1))


# In[38]:


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_axis.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))


# In[39]:


regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')
regressor.fit(X_axis, Y_axis, epochs = 100, batch_size = 32)


# In[41]:


test_data = pd.read_excel('NIFTY50_test.xlsx')
test_data


# In[43]:


real_prices = test_data['open'].values.reshape(-1,1)
real_prices


# In[46]:


input_test = pd.concat((data['open'],test_data['open']), axis=0)
input_test


# In[52]:


inputs = input_test[len(input_test)-len(test_data)-60:].values.reshape(-1,1)
inputs = scaler.transform(inputs)
inputs


# In[57]:


X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)    
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[58]:


predictions = regressor.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# In[59]:


plt.plot(real_prices, color = 'Red' , label="Actual Prices")
plt.plot(predictions, color = 'Blue', label="LSTM Predictions")
plt.xlabel("Time")
plt.ylabel("NIFTY50 opening price")
plt.legend()


# In[64]:


mse = mean_squared_error(real_prices, predictions)
np.sqrt(mse)


# In[68]:


r2  = r2_score(real_prices, predictions)
r2


# In[ ]:




