#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np


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


# # Summary Statistics for the overall data

# In[13]:


CreditCard2022.describe()


# # Total Amount For Each Category

# # Summary Statistics for all numerical and non-numeric columns

# ### Get the total amount of Auto/Transportation in Master Category

# In[14]:


Auto_Transportation_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Auto/Transportation', 'Amount'].sum()


# ### Print the total amount of Auto/Transportation

# In[15]:


print(f"Total amount of Auto/Transportation: {Auto_Transportation_sums}")


# ### Calculate Summary Statistics for Auto/Transportation

# In[16]:


Auto_Transportation_mean = np.mean(Auto_Transportation_sums)
Auto_Transportation_median = np.median(Auto_Transportation_sums)
Auto_Transportation_std = np.std(Auto_Transportation_sums)
Auto_Transportation_min = np.min(Auto_Transportation_sums)
Auto_Transportation_max = np.max(Auto_Transportation_sums)
Auto_Transportation_percentiles = np.percentile(Auto_Transportation_sums, [25, 50, 75])


# ### Print the summary statistics

# In[17]:


print(f"Mean: {Auto_Transportation_mean}")
print(f"Median: {Auto_Transportation_median}")
print(f"Standard deviation: {Auto_Transportation_std}")
print(f"Minimum: {Auto_Transportation_min}")
print(f"Maximum: {Auto_Transportation_max}")
print(f"25th, 50th, and 75th percentiles: {Auto_Transportation_percentiles}")


# ### Get the total amount of Bills/Utilities in Master Category

# In[18]:


Bills_Utilities_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Bills/Utilities', 'Amount'].sum()


# ### Print the total amount of Bills/Utilities

# In[19]:


print(f"Total amount of : {Bills_Utilities_sums}")


# ### Calculate Summary Statistics for Bills/Utilities

# In[20]:


Bills_Utilities_mean = np.mean(Bills_Utilities_sums)
Bills_Utilities_median = np.median(Bills_Utilities_sums)
Bills_Utilities_std = np.std(Bills_Utilities_sums)
Bills_Utilities_min = np.min(Bills_Utilities_sums)
Bills_Utilities_max = np.max(Bills_Utilities_sums)
Bills_Utilities_percentiles = np.percentile(Bills_Utilities_sums, [25, 50, 75])


# ### Print the summary statistics

# In[21]:


print(f"Mean: {Bills_Utilities_mean}")
print(f"Median: {Bills_Utilities_median}")
print(f"Standard deviation: {Bills_Utilities_std}")
print(f"Minimum: {Bills_Utilities_min}")
print(f"Maximum: {Bills_Utilities_max}")
print(f"25th, 50th, and 75th percentiles: {Bills_Utilities_percentiles}")


# ### Get the total amount of Business/Office in Master Category

# In[22]:


Business_Office_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Business/Office', 'Amount'].sum()


# ### Print the total amount of Business/Office

# In[23]:


print(f"Total amount of : {Business_Office_sums}")


# ### Calculate Summary Statistics for Business/Office

# In[24]:


Business_Office_mean = np.mean(Business_Office_sums)
Business_Office_median = np.median(Business_Office_sums)
Business_Office_std = np.std(Business_Office_sums)
Business_Office_min = np.min(Business_Office_sums)
Business_Office_max = np.max(Business_Office_sums)
Business_Office_percentiles = np.percentile(Business_Office_sums, [25, 50, 75])


# ### Print the summary statistics

# In[25]:


print(f"Mean: {Business_Office_mean}")
print(f"Median: {Business_Office_median}")
print(f"Standard deviation: {Business_Office_std}")
print(f"Minimum: {Business_Office_min}")
print(f"Maximum: {Business_Office_max}")
print(f"25th, 50th, and 75th percentiles: {Business_Office_percentiles}")


# ### Get the total amount of Children in Master Category

# In[26]:


Children_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Children', 'Amount'].sum()


# ### Print the total amount of Children

# In[27]:


print(f"Total amount of Children : {Children_sums}")


# ### Calculate Summary Statistics for Children

# In[28]:


Children_mean = np.mean(Children_sums)
Children_median = np.median(Children_sums)
Children_std = np.std(Children_sums)
Children_min = np.min(Children_sums)
Children_max = np.max(Children_sums)
Children_percentiles = np.percentile(Children_sums, [25, 50, 75])


# ### Print the summary statistics

# In[29]:


print(f"Mean: {Children_mean}")
print(f"Median: {Children_median}")
print(f"Standard deviation: {Children_std}")
print(f"Minimum: {Children_min}")
print(f"Maximum: {Children_max}")
print(f"25th, 50th, and 75th percentiles: {Children_percentiles}")


# ### Get the total amount of Credit Card/Loan Payments in Master Category

# In[30]:


Credit_Card_Loan_Payments_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Credit Card/Loan Payments', 'Amount'].sum()


# ### Print the total amount of Credit Card/Loan Payments

# In[31]:


print(f"Total amount of Credit Card/Loan Payments : {Credit_Card_Loan_Payments_sums}")


# ### Calculate Summary Statistics for Credit Card/Loan Payments

# In[32]:


Credit_Card_Loan_Payments_mean = np.mean(Credit_Card_Loan_Payments_sums)
Credit_Card_Loan_Payments_median = np.median(Credit_Card_Loan_Payments_sums)
Credit_Card_Loan_Payments_std = np.std(Credit_Card_Loan_Payments_sums)
Credit_Card_Loan_Payments_min = np.min(Credit_Card_Loan_Payments_sums)
Credit_Card_Loan_Payments_max = np.max(Credit_Card_Loan_Payments_sums)
Credit_Card_Loan_Payments_percentiles = np.percentile(Credit_Card_Loan_Payments_sums, [25, 50, 75])


# ### Print the summary statistics

# In[33]:


print(f"Mean: {Credit_Card_Loan_Payments_mean}")
print(f"Median: {Credit_Card_Loan_Payments_median}")
print(f"Standard deviation: {Credit_Card_Loan_Payments_std}")
print(f"Minimum: {Credit_Card_Loan_Payments_min}")
print(f"Maximum: {Credit_Card_Loan_Payments_max}")
print(f"25th, 50th, and 75th percentiles: {Credit_Card_Loan_Payments_percentiles}")


# ### Get the total amount of Entertainment in Master Category

# In[34]:


Entertainment_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == ' Entertainment', 'Amount'].sum()


# ### Print the total amount of Entertainment

# In[35]:


print(f"Total amount of Entertainment : { Entertainment_sums}")


# ### Calculate Summary Statistics for Entertainment

# In[36]:


Entertainment_mean = np.mean(Entertainment_sums)
Entertainment_median = np.median(Entertainment_sums)
Entertainment_std = np.std(Entertainment_sums)
Entertainment_min = np.min(Entertainment_sums)
Entertainment_max = np.max(Entertainment_sums)
Entertainment_percentiles = np.percentile(Entertainment_sums, [25, 50, 75])


# ### Print the summary statistics

# In[37]:


print(f"Mean: {Entertainment_mean}")
print(f"Median: {Entertainment_median}")
print(f"Standard deviation: {Entertainment_std}")
print(f"Minimum: {Entertainment_min}")
print(f"Maximum: {Entertainment_max}")
print(f"25th, 50th, and 75th percentiles: {Entertainment_percentiles}")


# ### Get the total amount of Food/Drink in Master Category

# In[38]:


Food_Drink_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Food/Drink', 'Amount'].sum()


# ### Print the total amount of Food/Drink

# In[39]:


print(f"Total amount of Food/Drink : {Food_Drink_sums}")


# ### Calculate Summary Statistics for Food/Drink

# In[40]:


Food_Drink_mean = np.mean(Food_Drink_sums)
Food_Drink_median = np.median(Food_Drink_sums)
Food_Drink_std = np.std(Food_Drink_sums)
Food_Drink_min = np.min(Food_Drink_sums)
Food_Drink_max = np.max(Food_Drink_sums)
Food_Drink_percentiles = np.percentile(Food_Drink_sums, [25, 50, 75])


# ### Print the summary statistics

# In[41]:


print(f"Mean: {Food_Drink_mean}")
print(f"Median: {Food_Drink_median}")
print(f"Standard deviation: {Food_Drink_std}")
print(f"Minimum: {Food_Drink_min}")
print(f"Maximum: {Food_Drink_max}")
print(f"25th, 50th, and 75th percentiles: {Food_Drink_percentiles}")


# ### Get the total amount of Health in Master Category

# In[42]:


Health_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Health', 'Amount'].sum()


# ### Print the total amount of Health

# In[43]:


print(f"Total amount of Health : {Health_sums}")


# ### Calculate Summary Statistics for Health

# In[44]:


Health_mean = np.mean(Health_sums)
Health_median = np.median(Health_sums)
Health_std = np.std(Health_sums)
Health_min = np.min(Health_sums)
Health_max = np.max(Health_sums)
Health_percentiles = np.percentile(Health_sums, [25, 50, 75])


# ### Print the summary statistics

# In[45]:


print(f"Mean: {Health_mean}")
print(f"Median: {Health_median}")
print(f"Standard deviation: {Health_std}")
print(f"Minimum: {Health_min}")
print(f"Maximum: {Health_max}")
print(f"25th, 50th, and 75th percentiles: {Health_percentiles}")


# ### Get the total amount of Home in Master Category

# In[46]:


Home_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Home', 'Amount'].sum()


# ### Print the total amount of Home

# In[47]:


print(f"Total amount of Home : {Home_sums}")


# ### Calculate Summary Statistics for Home

# In[48]:


Home_mean = np.mean(Home_sums)
Home_median = np.median(Home_sums)
Home_std = np.std(Home_sums)
Home_min = np.min(Home_sums)
Home_max = np.max(Home_sums)
Home_percentiles = np.percentile(Home_sums, [25, 50, 75])


# ### Print the Summary Statistics

# In[49]:


print(f"Mean: {Home_mean}")
print(f"Median: {Home_median}")
print(f"Standard deviation: {Home_std}")
print(f"Minimum: {Home_min}")
print(f"Maximum: {Home_max}")
print(f"25th, 50th, and 75th percentiles: {Home_percentiles}")


# ### Get the total amount of Miscellaneous in Master Category

# In[50]:


Miscellaneous_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Miscellaneous', 'Amount'].sum()


# ### Print the total amount of Miscellaneous

# In[51]:


print(f"Total amount of Miscellaneous : {Miscellaneous_sums}")


# ### Calculate Summary Statistics for Miscellaneous

# In[52]:


Miscellaneous_mean = np.mean(Miscellaneous_sums)
Miscellaneous_median = np.median(Miscellaneous_sums)
Miscellaneous_std = np.std(Miscellaneous_sums)
Miscellaneous_min = np.min(Miscellaneous_sums)
Miscellaneous_max = np.max(Miscellaneous_sums)
Miscellaneous_percentiles = np.percentile(Miscellaneous_sums, [25, 50, 75])


# ### Print the Summary Statistics

# In[53]:


print(f"Mean: {Miscellaneous_mean}")
print(f"Median: {Miscellaneous_median}")
print(f"Standard deviation: {Miscellaneous_std}")
print(f"Minimum: {Miscellaneous_min}")
print(f"Maximum: {Miscellaneous_max}")
print(f"25th, 50th, and 75th percentiles: {Miscellaneous_percentiles}")


# ### Get the total amount of Personnal Care in Master Category

# In[54]:


Personnal_Care_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Personnal Care', 'Amount'].sum()


# ### Print the total amount of Personnal Care

# In[55]:


print(f"Total amount of Personnal Care : {Personnal_Care_sums}")


# ### Calculate Summary Statistics for Personnal Care

# In[56]:


Personnal_Care_mean = np.mean(Personnal_Care_sums)
Personnal_Care_median = np.median(Personnal_Care_sums)
Personnal_Care_std = np.std(Personnal_Care_sums)
Personnal_Care_min = np.min(Personnal_Care_sums)
Personnal_Care_max = np.max(Personnal_Care_sums)
Personnal_Care_percentiles = np.percentile(Personnal_Care_sums, [25, 50, 75])


# ### Print the Summary Statistics for Personnal Care

# In[57]:


print(f"Mean: {Personnal_Care_mean}")
print(f"Median: {Personnal_Care_median}")
print(f"Standard deviation: {Personnal_Care_std}")
print(f"Minimum: {Personnal_Care_min}")
print(f"Maximum: {Personnal_Care_max}")
print(f"25th, 50th, and 75th percentiles: {Personnal_Care_percentiles}")


# ### Get the total amount of Shopping in Master Category

# In[58]:


Shopping_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Shopping', 'Amount'].sum()


# ### Print the total amount of Shopping

# In[59]:


print(f"Total amount of Shopping : {Shopping_sums}")


# ### Calculate Summary Statistics for Shopping

# In[60]:


Shopping_mean = np.mean(Shopping_sums)
Shopping_median = np.median(Shopping_sums)
Shopping_std = np.std(Shopping_sums)
Shopping_min = np.min(Shopping_sums)
Shopping_max = np.max(Shopping_sums)
Shopping_percentiles = np.percentile(Shopping_sums, [25, 50, 75])


# ### Print the Summary Statistics for Shopping

# In[61]:


print(f"Mean: {Shopping_mean}")
print(f"Median: {Shopping_median}")
print(f"Standard deviation: {Shopping_std}")
print(f"Minimum: {Shopping_min}")
print(f"Maximum: {Shopping_max}")
print(f"25th, 50th, and 75th percentiles: {Shopping_percentiles}")


# ### Get the total amount of Travel in Master Category

# In[62]:


Travel_sums = CreditCard2022.loc[CreditCard2022['Master Category'] == 'Travel', 'Amount'].sum()


# ### Print the total amount of Travel

# In[63]:


print(f"Total amount of Travel : {Travel_sums}")


# ### Calculate Summary Statistics for Travel

# In[64]:


Travel_mean = np.mean(Travel_sums)
Travel_median = np.median(Travel_sums)
Travel_std = np.std(Travel_sums)
Travel_min = np.min(Travel_sums)
Travel_max = np.max(Travel_sums)
Travel_percentiles = np.percentile(Travel_sums, [25, 50, 75])


# ### Print the Summary Statistics for Travel

# In[65]:


print(f"Mean: {Travel_mean}")
print(f"Median: {Travel_median}")
print(f"Standard deviation: {Travel_std}")
print(f"Minimum: {Travel_min}")
print(f"Maximum: {Travel_max}")
print(f"25th, 50th, and 75th percentiles: {Travel_percentiles}")


# # Inflation Rate for Food/Drink

# In[66]:


current_price = 5.45
base_price = 26.19

inflation_rate = ((current_price - base_price) / base_price) * 100

print("The inflation rate is: {:.2f}%".format(inflation_rate))


# # Inflation Rate for Shopping

# In[67]:


current_price = 26.07
base_price = 313.53

inflation_rate = ((current_price - base_price) / base_price) * 100

print("The inflation rate is: {:.2f}%".format(inflation_rate))


# ### Deflation for Shopping Category

# # Inflation Rate for 2022

# In[68]:


current_price = 39
base_price = 313.53

inflation_rate = ((current_price - base_price) / base_price) * 100

print("The inflation rate is: {:.2f}%".format(inflation_rate))


# ### Deflation for the whole data
