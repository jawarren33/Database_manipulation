#!/usr/bin/env python
# coding: utf-8

# 

# ### Interpreting the Orders Dataset
# 
# For questions 1, 2, and 3, you will work with sales data from a marketplace with several stores. The attached CSV, `screening_exercise_orders_v202102.csv`, lists the customer, date, and dollar value of orders placed in 2017. The gender of each customer is also provided. Please do not excluded $0 orders. 

# ### Exercise 1
# 
# Assemble a dataframe with one row per customer and the following columns:
# * customer_id 
# * gender
# * most_recent_order_date 
# * order_count (number of orders placed by this customer)
# 
# Sort the dataframe by customer_id ascending and display the first 10 rows.

# In[155]:


import pandas as pd

#import csv file and drop unused columns

df_raw = pd.read_csv('screening_exercise_orders_v202102.csv')
df_orders = df_raw.drop(['value'], axis=1)

#to find most recent order date, convert date column to datetime
#group by customer id, then find max(most recent)

df_orders['date'] = pd.to_datetime(df_orders['date'], yearfirst=True)
df_orders['most_recent_date'] = df_orders.groupby('customer_id')['date'].transform('max')

#to find number of orders placed by each customer, group all customer ids
#then create a column that counts number of times customer id appears
df_orders['order_count'] = df_orders.groupby('customer_id')['date'].transform('count')
df = df_orders.drop(['date'], axis=1)

#sort by customer_id
df.sort_values(by='customer_id')

df.head(10)


# ### Exercise 2
# Plot the count of orders per week (for all stores together). Do not use plotly, as plotly graphs in Jupyter Notebooks don't render correctly on different machines.

# In[193]:


import seaborn as sns
import matplotlib.pyplot as plt

#group the data by week and sum them together. 
#stored in dataframe: df_plt
#plot the timeseries data using a line graph from seaborn library to show the sales trend over the year

df_plt = df_orders.resample('W-Mon', on = 'date').sum().drop(['customer_id','gender'],axis = 1)
line_plt = sns.lineplot(data = df_plt, palette = 'tab10', linewidth = 3 )


# ### Exercise 3
# 
# Compute the mean order value for gender 0 and for gender 1. Do you think the difference is significant? 
# 
# **Justify your choice of method and state any assumptions you make. Make sure you are clear about why your method is suitable for this dataset, in particular.**.

# In[175]:


print('THE MEAN FOR EACH GENDER IS: \n', df_raw.groupby(['gender'])['value'].mean())
print('\n')
print('THE STANDARD DEVIATION FOR VALUE IS: \n', df_raw['value'].std(),'\n')

#The difference in purchase value between each gender is not significant as it is much
#less than the standard deviation from the mean of puchase value

#This is the correct method to use as it requires the least lines of code, and assessing the significance in comparison
#to the standard deviation is the most logical way to determine statistical significance


# ### Interpreting the Product Dataset
# For this question you will work with the data in the CSV `screening_exercise_products_v202102.csv`. This dataset shows the number of times a particular item was sold. It contains the following columns: 
# * number_of_orders - the number of orders containing the product
# * store_id - the ID of the store selling the product
# * product_id - the ID of the product itself
# * is_red - whether the product is red.

# ### Exercise 4
# 
# Suppose some of our customers came to us with the belief that the color red generates more sales than other colors.  Based on this dataset, would you suggest that companies color more of their products red? Why or why not, and what other factors could be important to determine this? 
# 
# **Please justify your answer and state any assumptions you make.**

# In[192]:


df_prod = pd.read_csv('screening_exercise_products_v202102.csv')

print('THE AVERAGE NUMBER OF ORDERS IF THE COLOR IS NOT RED/IS RED IS: \n', df_prod.groupby(['is_red'])['number_of_orders'].mean())
print('\n')
print('')
print('THE STANDARD DEVIATION OF ORDERS FROM THE MEAN FOR THE DATASET IS: \n', df_prod['number_of_orders'].std(), '\n')


#As you can see below, the number of orders is greater for items that are colored red. However, the difference is still
#below the standard deviation for all orders. This means that the difference is not statistically significant and I 
#would not recommend that companies color more of their products red based on this information only. There may be other 
#factors that effect this outcome such as certain red products assessed are just higher in demand based on utility alone,
#and not the color of the product. An interesting experiment would be to make the same product different colors, including
#red, and seeing which color produces the most sales.


# ### Exercise 5
# 
# Describe one of your favorite tools or techniques and give a small example of how it has helped you solve a problem. Limit your answer to one short paragraph, and please be specific. 

# In[ ]:


#One of my favorite techniques to use is linear regression for prediction modelling. I have been using it since my
#econometrics classes in undergraduate school to predict housing prices based on different demographics. I recently 
#used it for a hobby of mine,fantasy sports. I created a model to predict the probability of players scoring a 
#certain number of points in a game.I included about 10 different variables as to not overfit the model. 
#I used the SKLearn library for the linear regression. I love using that library for linear regression as it 
#can train and validate the model in minimal lines of code. Using the R-squared value, it is also easy to see how well
#the data fits to the model.

