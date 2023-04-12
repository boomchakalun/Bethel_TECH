#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# # Total Amount For Each Category

# ### Get the total amount of Auto/Transportation in Master Category

# In[14]:


Auto_Transportation_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Auto/Transportation', 'Amount'].sum()


# ### Print the total amount of Auto/Transportation

# In[15]:


print(f"Total amount of Auto/Transportation: {Auto_Transportation_sums}")


# ### Get the total amount of Bills/Utilities in Master Category

# In[16]:


Bills_Utilities_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Bills/Utilities', 'Amount'].sum()


# ### Print the total amount of Bills/Utilities

# In[17]:


print(f"Total amount of : {Bills_Utilities_sums}")


# ### Get the total amount of Business/Office in Master Category

# In[18]:


Business_Office_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Business/Office', 'Amount'].sum()


# ### Print the total amount of Business/Office

# In[19]:


print(f"Total amount of : {Business_Office_sums}")


# ### Get the total amount of Children in Master Category

# In[20]:


Children_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Children', 'Amount'].sum()


# ### Print the total amount of Children

# In[21]:


print(f"Total amount of Children : {Children_sums}")


# ### Get the total amount of Credit Card/Loan Payments in Master Category

# In[22]:


Credit_Card_Loan_Payments_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Credit Card/Loan Payments', 'Amount'].sum()


# ### Print the total amount of Credit Card/Loan Payments

# In[23]:


print(f"Total amount of Credit Card/Loan Payments : {Credit_Card_Loan_Payments_sums}")


# ### Get the total amount of Entertainment in Master Category

# In[24]:


Entertainment_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == ' Entertainment', 'Amount'].sum()


# ### Print the total amount of Entertainment

# In[25]:


print(f"Total amount of Entertainment : { Entertainment_sums}")


# ### Get the total amount of Food/Drink in Master Category

# In[26]:


Food_Drink_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Food/Drink', 'Amount'].sum()


# ### Print the total amount of Food/Drink

# In[27]:


print(f"Total amount of Food/Drink : {Food_Drink_sums}")


# ### Get the total amount of Health in Master Category

# In[28]:


Health_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Health', 'Amount'].sum()


# ### Print the total amount of Health

# In[29]:


print(f"Total amount of Health : {Health_sums}")


# ### Get the total amount of Home in Master Category

# In[30]:


Home_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Home', 'Amount'].sum()


# ### Print the total amount of Home

# In[31]:


print(f"Total amount of Home : {Home_sums}")


# ### Get the total amount of Miscellaneous in Master Category

# In[32]:


Miscellaneous_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Miscellaneous', 'Amount'].sum()


# ### Print the total amount of Miscellaneous

# In[33]:


print(f"Total amount of Miscellaneous : {Miscellaneous_sums}")


# ### Get the total amount of Personnal Care in Master Category

# In[34]:


Personnal_Care_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Personnal Care', 'Amount'].sum()


# ### Print the total amount of Personnal Care

# In[35]:


print(f"Total amount of Personnal Care : {Personnal_Care_sums}")


# ### Get the total amount of Shopping in Master Category

# In[36]:


Shopping_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Shopping', 'Amount'].sum()


# ### Print the total amount of Shopping

# In[37]:


print(f"Total amount of Shopping : {Shopping_sums}")


# ### Get the total amount of Travel in Master Category

# In[38]:


Travel_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Travel', 'Amount'].sum()


# ### Print the total amount of Travel

# In[39]:


print(f"Total amount of Travel : {Travel_sums}")

