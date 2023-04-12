#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import geopandas as gpd
import folium
import seaborn as sns
import warnings


# # Load in Data

# ## The data was filter from Wells Fargo personal credit card bank statement.

# ## This data is filter only for the year of 2022 through the csv file in Excel.

# In[2]:


CreditCard2022 = pd.read_csv('C:/Users/Cing San Lun/Desktop/Bethel/DSO110 - Final Group Project/Lun, Cing_DSO110-final-Project/Data-Clean/Credit Card Year End 2022.csv')
CreditCard2022.head()


# In[3]:


CreditCard2022.tail()


# # Check the shape of the data

# In[4]:


CreditCard2022.shape


# # Remove 'Payment Method' Column

# In[5]:


CreditCard2022.drop('Payment Method', axis=1, inplace=True)


# In[6]:


CreditCard2022.head()


# # Split the Date column into Month, Day, and Year column

# In[7]:


CreditCard2022[['Month', 'Day', 'Year']] = CreditCard2022['Date'].str.split("/", expand = True)


# ## Convert the 'Year' column to integer format

# In[8]:


CreditCard2022 = CreditCard2022.astype({'Year':'int'})


# In[9]:


CreditCard2022.head()


# # Convert the 'Amount' column to currency format

# ## I can now do math operation specifically on 'Amount column'

# In[10]:


CreditCard2022['Amount'] = CreditCard2022['Amount'].replace('[^\d.]', '', regex=True).astype(float)


# In[11]:


CreditCard2022.info()


# # Check the overall info

# In[12]:


CreditCard2022.info()


# In[13]:


CreditCard2022.describe()


# # Pie Chart

# ## Percentage by Master Category

# In[14]:


fig = px.pie(CreditCard2022['Month'], CreditCard2022['Master Category'])
fig.show()


# # Bar Graph

# ## Spending By Category

# In[15]:


fig = px.bar(CreditCard2022, x='Amount', y='Master Category', orientation='h', height=400)
fig.show()


# In[16]:


fig = px.bar(CreditCard2022, x='Amount', y='Master Category', color='Amount', orientation='h', height=400)
fig.show()


# ## Average spend by Date

# In[17]:


fig = px.bar(CreditCard2022, x='Date', y='Amount')
fig.show()


# In[18]:


fig = px.bar(CreditCard2022, x='Date', y='Amount', color='Amount')
fig.show()


# # Boxplot

# ### Create a boxplot for Amount

# In[19]:


fig = px.box(CreditCard2022, x='Amount')
fig.show()


# ### Create a boxplot for Master Category

# In[20]:


fig = px.box(CreditCard2022, x='Master Category', y='Amount')
fig.show()


# In[21]:


fig = px.box(CreditCard2022, x='Master Category', y='Amount', color='Master Category')
fig.show()


# # Scatterplot

# ### Create a scatterplot for Master Category and Amount

# In[22]:


fig = px.scatter(CreditCard2022, x='Master Category', y='Amount')
fig.show()


# In[23]:


fig = px.scatter(CreditCard2022, x='Master Category', y='Amount', color='Master Category')
fig.show()


# # Map

# ## Show what Location I use my credit card at the most

# ### Creat an object of the map

# In[24]:


m = folium.Map(location=[26.7153, -80.0534], zoom_start=12)


# ### Add a marker

# In[25]:


folium.Marker(
    location=[26.7153, -80.0534],
    popup="West Palm Beach",
).add_to(m)


# ### Save the map to an HTML file

# In[26]:


m.save('map.html')

