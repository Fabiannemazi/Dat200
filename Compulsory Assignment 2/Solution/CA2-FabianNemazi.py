#!/usr/bin/env python
# coding: utf-8

# In[367]:


cd


# In[368]:


cd fag_nmbu/Dat200/CA2/Data


# ### Importing the nessescary packages

# In[372]:


import pandas as pd
import seaborn as sns; sns.set(context = 'paper', style = 'whitegrid')
import matplotlib.pyplot as plt
import numpy as np
from perceptron_ext import Perceptron
#from adaline_ext import AdalineGD


# ### Creating the column names for the dataframe

# In[182]:


column_names = ['Number_of_times_pregnant', 'Plasma_glucose_concentration',
               'Diastolic_blood_pressure', 'Triceps_skin fold_thickness', '2-Hour_serum_insulin',
               'Body_mass_index', 'Diabetes_pedigree_function', 'Age', 'Class_variable']


# In[183]:


df = pd.read_csv('CA2_data.csv', names = column_names, header = None)


# ### Task 1 : Visualize the raw data

# In[382]:


sns.pairplot(df, hue='Class_variable')


# At first glance it may look like we don't have to classes that are linearly seperable. For a model like Perceptron, where convergence of to classes are only guaranteeded if they are linearly seperable, this may be hard. It may be a reasonable guess to think that the Adaline model would to better on this dataset.

# ### Task 2: removing all suspsect pasients

# In[374]:


indexing = df.query('Diastolic_blood_pressure == 0 | Body_mass_index == 0').index.tolist() #using df.query to pick out the columns i want
df = df.drop(indexing).reset_index(drop = True) # resetting the indexing because we have removed values


# ### Creating functions for splitting and standarizing data

# In[375]:


def scale_features_data(df, X_train, X_test):
    """
    From lecture: standardizes our data-subsets.
    """
    X_train_std = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=0)
    X_test_std = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=0)   
    
    return X_train_std, X_test_std


# In[376]:


def split_dataset(df, num_training_data):
    """
    Splits our dataset into the desired subsets for training and testing.
    """
    X_train = df.iloc[:num_training_data].drop('Class_variable', axis=1).values
    X_test = df.iloc[num_training_data:].drop('Class_variable', axis=1).values
    
    y_train = df.iloc[:num_training_data, 8].values
    y_test = df.iloc[num_training_data:, 8].values
    
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    
    return X_train, X_test, y_train, y_test


# In[377]:


#training_set = df.loc[:399, :].values
#test_set = df.loc[400:, :].values


# In[232]:


#X_train = training_set[:, :8].values
#y_train = np.where(training_set[:, -1] == 0, -1, 1)
#X_test = test_set[:, :8]
#y_test = np.where(test_set[:, -1] == 0, -1, 1)


# ### Creating column names and index names for use later

# In[149]:


Perceptron_columns = ['Epoch ' + str(name)  for name in range(1,51)]
Perceptron_index = [str(index) + ' out of 400' for index in range(50, 450, 50)]


# ### Task 3: training the 400 models, computing classification accuracy and storing them in a dataframe

# ### Perceptron algorithm

# In[378]:


df_Perceptron = pd.DataFrame(columns =range(1,51), index=range(50, 450, 50)) #creating df

for column in list(range(1,51)):
    df_Perceptron[column] = df_Perceptron[column].astype(float)
    #for a weird reason the heatmap plotting woldnt work unless the values was float, so that is the reason
    #for me doing this
    
    
    
for nrows in range(50, 450, 50):
    for epoch in range(1,51):
        X_train, X_test, y_train, y_test = split_dataset(df, nrows) #splitting the dataset with the func created
        X_train_std, X_test_std = scale_features_data(df, X_train, X_test) #Standarizing the datasets
        ppn = Perceptron(eta=0.0001, n_iter=epoch) #Creating Perceptron object    
        #train model here
        ppn.fit(X_train_std, y_train)
        # predict
        y_pred = ppn.predict(X_test_std)
        
        
        # find accuracy
        accuracy = (y_test == y_pred).sum() / len(y_test) * 100
        df_Perceptron.loc[nrows, epoch] = accuracy
        
        
df_Perceptron.columns = Perceptron_columns
df_Perceptron.index = Perceptron_index


# ### Doing the same just with adaline algorithm

# In[299]:


df_Adaline = pd.DataFrame(columns =range(1,51), index= range(50, 450, 50) )
Adaline_columns = ['Epoch ' + str(name)  for name in range(1,51)]
Adaline_index = [str(index) + ' out of 400' for index in range(50, 450, 50)]


# In[300]:


for column in range(1,51):
    df_Adaline[column] = df_Adaline[column].astype(float)
for nrows in range(50, 450, 50):
    for epoch in range(1,51):
        X_train, X_test, y_train, y_test = split_dataset(df, nrows)
        X_train_std, X_test_std = scale_features_data(df, X_train, X_test)
        ada = AdalineGD(eta=0.0001, n_iter=epoch)    
        #train model here
        ada.fit(X_train_std, y_train)
        # predict
        y_pred = ada.predict(X_test_std)
        
        
        # find accuracy
        accuracy = (y_test == y_pred).sum() / len(y_test) * 100
        df_Adaline.loc[nrows, epoch] = accuracy
        
df_Adaline.columns = Adaline_columns
df_Adaline.index = Adaline_index


# ### Task 4: Plotting the heatmap of Perceptron

# In[270]:


plt.subplots(figsize=(18,13))
ax = sns.heatmap(df_Perceptron, vmax=90)


# ### Plotting the heatmap of Adaline

# In[304]:


plt.subplots(figsize=(18,13))
ax = sns.heatmap(df_Adaline, vmax=90)


# ### Task 5: maximum test set classification accuracy

# ### Perceptron maximum

# In[381]:


highest_value_ppn = max(list(df_Perceptron.max())) #storing the highest value found in the dataset
epoch_iteration_ppn = df_Perceptron.max().idxmax() #storing the epoch that was assosciated with highest value
index_value_ppn = df_Perceptron[epoch_iteration].idxmax() #Getting the index for that value as well
print(f'For Perceptron the highest accuracy value was: {highest_value_ppn} \nThe epoch was: {epoch_iteration_ppn} \nThe size of the dataset was: {index_value_ppn}')


# ### Adaline Maximum

# In[380]:


highest_value_ada = max(list(df_Adaline.max())) #storing the highest value found in the dataset
epoch_iteration_ada = df_Adaline.max().idxmax() #storing the epoch that was assosciated with highest value
index_value_ada = df_Adaline[epoch_iteration].idxmax() #Getting the index for that value as well
print(f'For Perceptron the highest accuracy value was: {highest_value_ada} \nThe epoch was: {epoch_iteration_ada} \nThe size of the dataset was: {index_value_ada}')


# # TASK 6: training time

# The reason for the longer simulation time in Perceptron versus Adaline is that during the Perceptron model training it updates the weight immediately after a misclassification. So during a specific epoch, it can update the weight multiple times. The Adaline only updates at the end of each Epoch. So in a big simulation this will be considerably more time confusing for Perceptron. 

# In[ ]:




