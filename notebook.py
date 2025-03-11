#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis - Assignment
# 
# ## 🔍 Overview
# This lab is designed to help you practice exploratory data analysis using Python. You will work with some housing data for the state of California. You will use various data visualization and analysis techniques to gain insights and identify patterns in the data, and clean and preprocess the data to make it more suitable for analysis. The lab is divided into the following sections:
# 
# - Data Loading and Preparation
# - Data Visualization
# - Data Cleaning and Preprocessing (using visualizations)
# 
# ## 🎯 Objectives
# This assignment assess your ability to:
# - Load and pre-process data using `pandas`
# - Clean data and preparing it for analysis
# - Use visualization techniques to explore and understand the data
# - Use visualization techniques to identify patterns and relationships in the data
# - Use visualization to derive insights from the data
# - Apply basic statistical analysis to derive insights from the data
# - Communicate your findings through clear and effective data visualizations and summaries

# #### Package Imports
# We will keep coming back to this cell to add "import" statements, and configure libraries as we need

# In[1]:


# Common imports
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

# To plot figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
# Configure pandas to display 500 rows; otherwise it will truncate the output

pd.set_option('display.max_rows', 500)
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")


# ## Housing Data in California

# ### Task 1:  Load the dataset
# The dataset is available in the `data/housing.csv` file. Check the file to determine the delimiter and/or the appropriate pandas method to use to load the data.
# 
# Make sure you name the variable `housing` and that you use the appropriate pandas method to load the data.

# In[3]:


# 💻 Import the dataset in the project (data/housing.csv) into a dataframe called (housing)
import os
housing = pd.read_csv('data/housing.csv')
print(os.path.exists('data/housing.csv'))




# ### Task 2: Confirm the data was loaded correctly

# #### 2.1: Get the first 6 records of the dataset

# In[4]:


# 💻 Get the first 6 records of the dataframe
housing.head(6)


# #### 2.2: Get the last 7 records of the dataset

# In[5]:


# 💻 Get the last 7 records of the dataframe
housing.tail(7)


# #### 2.3: Get a random sample of 10 records

# In[6]:


# 💻 Get a random 10 records of the dataframe
housing.sample(10)


# #### 2.4: Get information about the dataset, including the number of rows, number of columns, column names, and data types of each column

# In[7]:


# 💻 Show information about the different data columns (columns, data types, ...etc.)
housing.info()


# > 🚩 This is a good point to commit your code to your repository.

# ### Task 3: Understand the data types
# For each of the 10 columns, Identify the data type: (Numerical-Continuous, Numerical-Discrete, Categorical-Ordinal, Categorical-nominal )
# 
# <details>
# <summary>Click here for the data type diagram</summary>
# 
#   ![Data types](https://miro.medium.com/max/1400/1*kySPZcf83qLOuaqB1vJxlg.jpeg)
# </details>
Longitude:          💻: Numerical-Continuous
Latitude:           💻: Numerical-Continuous
Housing Median Age: 💻: Numerical-Discrete
Total Rooms:        💻: Numerical-Discrete
Total Bedrooms:     💻: Numerical-Discrete
Population:         💻: Numerical-Discrete
Households:         💻: Numerical-Discrete
Median Income:      💻: Numerical-Continuous
Median House Value: 💻: Numerical-Continuous
Ocean Proximity:    💻: Catigorical-Nominal
# > 🚩 This is a good point to commit your code to your repository.

# ### Task 4: Understand the data
# #### 4.1: Get the summary statistics for the numerical columns

# In[8]:


# 💻 Show the descriptive statistics information about the columns in the data frame
housing.describe()


# #### 4.2: For the categorical columns, get the frequency counts for each category
# 
# <details>
#   <summary>🦉 Hints</summary>
# 
#   - Use the `value_counts()` method on the categorical columns
# </details>

# In[9]:


# 💻 Show the frequency of the values in the ocean_proximity column
housing.value_counts('ocean_proximity')


# > 🚩 This is a good point to commit your code to your repository.

# ### Task 5: Visualize the data

# #### 5.1: Visualize the distribution of the numerical columns
# In a single figure, plot the histograms for all the numerical columns. Use a bin size of 50 for the histograms

# In[16]:


# 💻 Plot a histogram of all the data features (with a bin size of 50)
#housing.hist(bins=50, figsize=(20,15), layout=(3, 3))
#plt.tight_layout()
#plt.show()
housing_data=housing.select_dtypes(include=['number'])
plt.figure(figsize=(20,15))
plt.hist(housing_data.values.flatten(), bins=50, alpha=0.5)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Numerical Columns")
plt.show()


# #### 5.2: Visualize the distribution of only one column
# Plot the histogram for the `median_income` column. Use a bin size of 50 for the histogram

# In[11]:


# 💻 plot a histogram of only the median_income
housing['median_income'].hist(bins=50, figsize=(20,15))
plt.xlabel('Median Income')
plt.ylabel('Frequency')
plt.title('Median Income Histogram')
plt.show()


# > 🚩 This is a good point to commit your code to your repository.

# #### 5.3: Visualize the location of the houses using a scatter plot
# In a single figure, plot a scatter plot of the `longitude` and `latitude` columns. 
# 
# 
# Try this twice, once setting the `alpha` parameter to set the transparency of the points to 0.1, and once without setting the `alpha` parameter.

# In[17]:


# 💻 scatter plat without alpha
plt.figure(figsize=(20,15))
plt.scatter(housing['longitude'], housing['latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Longitude vs Latitude')
plt.show()


# In[18]:


# 💻 scatter plat with alpha
plt.figure(figsize=(20,15))
plt.scatter(housing['longitude'], housing['latitude'], alpha=0.1) 
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Longitude vs Latitude')
plt.show()


# > 🚩 This is a good point to commit your code to your repository.

# 💯✨ For 3 Extra Credit points; Use the Plotly express to plot the scatter plot on a map of california
# 
# (📜 Check out the examples on their docs)[https://plotly.com/python/scatter-plots-on-maps/]

# In[31]:


# 💻💯✨ Plot the data on a map of California

import plotly.express as px
fig = px.scatter_map(housing, lat="latitude", lon="longitude", hover_name="median_house_value", hover_data=["ocean_proximity", "median_income"], size="population", zoom=6, height=600)
fig.update_layout(map_style="open-street-map")
fig.show()


# > 🚩 This is a good point to commit your code to your repository.

# ### Task 6: Explore the data and find correlations

# #### 6.1: Generate a correlation matrix for the numerical columns

# In[ ]:


# 💻 Get the correlation matrix of the housing data
import seaborn as sns
corr_matrix = housing_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# #### 6.2: Get the Correlation data fro the `median_house_age` column
# sort the results in descending order

# In[37]:


# 💻 Get the correlation data for just the housing_median_age
df = pd.DataFrame(housing_data['housing_median_age'])
corr_matrix = df.corr()
housing_median_age_corr = corr_matrix['housing_median_age']
print(housing_median_age_corr)


# #### 6.2: Visualize the correlation matrix using a heatmap
# - use the coolwarm color map
# - show the numbers on the heatmap
# 

# In[39]:


# 💻 Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(housing_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# #### 6.3: Visualize the correlations between some of the features using a scatter matrix
# - Plot a scatter matrix for the `total_rooms`, `median_house_age`, `median_income`, and `median_house_value` columns

# In[46]:


# 💻 using Pandas Scatter Matrix Plotting, Plot the scatter matrix for (median_house_value, median_income, total_rooms, housing_median_age)
scatter_matrix(housing[['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']], figsize=(12, 8))
plt.show()


# #### 6.4: Visualize the correlations between 2 features using a scatter plot
# - use an `alpha` value of 0.1

# In[44]:


# 💻 Plot the scatter plot for just (median_income and median_house_value)
feature_x = 'median_income'
feature_y = 'median_house_value'
plt.figure(figsize=(10, 6))
plt.scatter(housing[feature_x], housing[feature_y], alpha=0.1)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Scatter Plot of Median Income vs Median House Value')
plt.show()


# #### 6.5: ❓ What do you notice about the chart? what could that mean?
# What could the lines of values at the top of the chart mean here?
💻: This could possibly represent the houses that have a value above $500,000, since the cap is $500,000 for this particular graph
# > 🚩 This is a good point to commit your code to your repository.

# ### Task 7: Data Cleaning - Duplicate Data

# #### 7.1: Find duplicate data

# In[47]:


# 💻 Identify the duplicate data in the dataset
duplicates = housing.duplicated()
print("Number of duplicate rows:", duplicates.sum())


# ### Task 8: Data Cleaning - Missing Data

# #### 8.1: Find missing data

# In[48]:


# 💻 Identify the missing data in the dataset
missing_data = housing.isnull().sum()
print("Missing data in each column:")
print(missing_data)


# #### 8.2: show a sample of 5 records of the rows with missing data
# Notice there are 2 keywords here: `sample` and (rows with missing data)
# 
# <details>
#   <summary>🦉 Hints:</summary>
# 
#   * You'll do pandas filtering here
#   * You'll need to use the `isna()` or `isnull()` method on the 1 feature with missing data. to find the rows with missing data
#   * you'll need to use the `sample()` method to get a sample of 5 records of the results
# </details>

# In[53]:


# 💻 use Pandas Filtering to show all the records with missing `total_bedrooms` field
total_bedrooms_missing = housing[housing['total_bedrooms'].isnull()]
sample_missing = total_bedrooms_missing.sample(5)
print("Total_bedrooms missing data:")
print(sample_missing)


# #### 8.3: Calculate the central tendency values of the missing data feature
# * Calculate the mean, median, trimmed mean

# In[55]:


# 💻 get the mean, median and trimmed mean of the total_bedrooms column
#total_bedrooms_median = 0
#total_berooms_mean = 0
#total_bedrooms_trimmed_mean = 0
mean_value = housing['total_bedrooms'].mean()
median_value = housing['total_bedrooms'].median()
trimmed_mean_value = trim_mean(housing['total_bedrooms'].dropna(), proportiontocut=0.1)
print(f"Median: {median_value}")
print(f"Mean: {mean_value}")
print(f"Trimmed Mean: {trimmed_mean_value}")


# #### 8.4: Visualize the distribution of the missing data feature
# * Plot a histogram of the missing data feature (total_bedrooms)

# In[56]:


# 💻 Plot the histogram of the total_bedrooms column
df = pd.DataFrame(housing['total_bedrooms'])
plt.figure(figsize=(10, 6))
plt.hist(df['total_bedrooms'].dropna(), bins=50, alpha=0.7)
plt.xlabel('Total Bedrooms')
plt.ylabel('Frequency')
plt.title('Histogram of Total Bedrooms')
plt.show()


# #### 8.5: Choose one of the central tendency values and use it to fill in the missing data
# * Justify your choice
# * Don't use the `inplace` parameter, instead, create a new dataframe with the updated values. (this is a bit challenging)
# * show the first 5 records of the new dataframe to confirm we got the full dataframe
# 
# [📜 You should find a good example here](https://www.sharpsightlabs.com/blog/pandas-fillna/#example-2)

# In[57]:


# 💻 Fill the missing values in the total_bedrooms column with an appropriate value, then show the first 5 records of the new dataframe
mean_value = housing['total_bedrooms'].mean()
housing['total_bedrooms'].fillna(mean_value, inplace=True)
print("First 5 records after filling missing values:")
print(housing.head())


# ❓ Why did you choose this value?
💻 I chose the mean because its not heavily skewed and shows a good representation of the average.
# #### 8.6: Confirm that there are no more missing values in the new dataframe
# * make sure the dataframe contains all features, not just the `total_bedrooms` feature

# In[58]:


# 💻 Confirm the new dataframe has no missing values
missing_values = housing.isnull().sum()
print("Missing values after filling:")
print(missing_values)


# #### 8.7: Dropping the missing data
# assume we didn't want to impute the missing data, and instead, we wanted to drop the rows with missing data.
# * don't use the `inplace` parameter, instead, create a new dataframe with the updated values.

# In[59]:


# 💻 drop the missing rows of the total_bedroom and save it to a new dataframe
missing_values_dropped = housing.dropna(subset=['total_bedrooms'])
print("DataFrame after dropping missing values:")
print(missing_values_dropped.head())


# #### 8.8: Confirm that there are no more missing values in the new dataframe
# * make sure the dataframe contains all features, not just the `total_bedrooms` feature

# In[62]:


# 💻 Confirm the new dataframe has no missing values
missing_values_dropped = missing_values_dropped.isnull().sum()      
print(missing_values_dropped)


# > 🚩 This is a good point to commit your code to your repository.

# ## Wrap up
# Remember to update the self reflection and self evaluations on the `README` file.

# Make sure you run the following cell; this converts this Jupyter notebook to a Python script. and will make the process of reviewing your code on GitHub easier

# In[63]:


# 🦉: The following command converts this Jupyter notebook to a Python script.
get_ipython().system('jupyter nbconvert --to python notebook.ipynb')


# > 🚩 **Make sure** you save the notebook and make one final commit here
