#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
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


# # Summary Statistics for the overall data

# In[13]:


CreditCard2022.describe()


# # Correlation

# ### Define the category-to-integer mapping

# In[14]:


master_category_map = {
    'Auto/Transportation': 1,
    'Bills/Utilities': 2,
    'Business/Office': 3,
    'Children': 4,
    'Credit Card/Loan Payments': 5,
    'Entertainment': 6,
    'Food/Drink': 7,
    'Health': 8,
    'Home': 9,
    'Miscellaneous': 10,
    'Personnal Care': 11,
    'Shopping': 12,
    'Travel': 13,
}


# ### Convert a category string to an integer

# In[15]:


CreditCard2022['master_category_int'] = CreditCard2022['Master Category'].map(master_category_map)


# ### Calculate correlation between category and amount

# In[16]:


corr = CreditCard2022['master_category_int'].corr(CreditCard2022['Amount'])


# In[17]:


print(f"The correlation between master category and amount is: {corr:.2f}")


# ### There is a weak negative correlation between Master Category and Amount. 
# ### As one variable increases, the other variable tends to decreases slightly, but this relationship is not very strong.

# # ANOVA

# ### Separate the data by category

# In[18]:


categories = {}
for Auto_Transportation in CreditCard2022['Master Category'].unique():
    categories[Auto_Transportation] = CreditCard2022[CreditCard2022['Master Category'] == Auto_Transportation]['Amount']
    
for Bills_Utilities in CreditCard2022['Master Category'].unique():
    categories[Bills_Utilities] = CreditCard2022[CreditCard2022['Master Category'] == Bills_Utilities]['Amount']
    
for Business_Office in CreditCard2022['Master Category'].unique():
    categories[Business_Office] = CreditCard2022[CreditCard2022['Master Category'] == Business_Office]['Amount']
    
for Children in CreditCard2022['Master Category'].unique():
    categories[Children] = CreditCard2022[CreditCard2022['Master Category'] == Children]['Amount']
    
for Credit_Card_Loan_Payments in CreditCard2022['Master Category'].unique():
    categories[Credit_Card_Loan_Payments] = CreditCard2022[CreditCard2022['Master Category'] == Credit_Card_Loan_Payments]['Amount']
    
for Entertainment in CreditCard2022['Master Category'].unique():
    categories[Entertainment] = CreditCard2022[CreditCard2022['Master Category'] == Entertainment]['Amount']
    
for Food_Drink in CreditCard2022['Master Category'].unique():
    categories[Food_Drink] = CreditCard2022[CreditCard2022['Master Category'] == Food_Drink]['Amount']
    
for Health in CreditCard2022['Master Category'].unique():
    categories[Health] = CreditCard2022[CreditCard2022['Master Category'] == Health]['Amount']
    
for Home in CreditCard2022['Master Category'].unique():
    categories[Home ] = CreditCard2022[CreditCard2022['Master Category'] == Home ]['Amount']

for Miscellaneous in CreditCard2022['Master Category'].unique():
    categories[Miscellaneous] = CreditCard2022[CreditCard2022['Master Category'] == Miscellaneous]['Amount']
    
for Personnal_Care in CreditCard2022['Master Category'].unique():
    categories[Personnal_Care] = CreditCard2022[CreditCard2022['Master Category'] == Personnal_Care]['Amount']
    
for Shopping in CreditCard2022['Master Category'].unique():
    categories[Shopping] = CreditCard2022[CreditCard2022['Master Category'] == Shopping]['Amount']
    
for Travel in CreditCard2022['Master Category'].unique():
    categories[Travel] = CreditCard2022[CreditCard2022['Master Category'] == Travel]['Amount']


# ### Perform ANOVA test

# In[19]:


f_stat, p_value = f_oneway(*categories.values())


# ### Print the results

# In[21]:


print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_value:.2f}")


# In[39]:


model = ols("Amount ~ master_category_int", data=CreditCard2022).fit()
anova_table = sm.stats.anova_lm(model, typ=3)


# ### The results are statistically significant and provide strong evidence against the null hypothesis.

# ### Print degrees of freedom and significance level

# In[40]:


df_between = anova_table["df"][0]
df_within = anova_table["df"][1]
df_total = anova_table["df"][2]
sig_level = 0.05


# In[41]:


print("Degrees of freedom for between-group variation:", df_between)
print("Degrees of freedom for within-group variation:", df_within)
print("Degrees of freedom for total variation:", df_total)
print("Significance level:", sig_level)


# # Sample Size

# In[24]:


len(CreditCard2022)


# ### p-value less than or equal to the significance level (0.05 in this case) indicates that we reject the null hypothesis and conclude that there is a significant difference between the groups. 
# 
# ### We reject the null hypothesis and conclude that there is a significant difference between the groups.
