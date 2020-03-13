import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set(context = 'paper', style = 'whitegrid')
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from pprint import pprint

#Part 1
A = pd.read_excel("A.xlsx")
B = pd.read_excel("B.xlsx")
C = pd.read_excel("C.xlsx")
D = pd.read_excel("D.xlsx")
dataset = {"A" : A, "B" : B, "C" : C, "D" : D} 

# Part 2
for x in dataset:
    sets = dataset[x]
    if sets.isna().values.any(): #using pandas method chaining
        print(f'Missing values in dataframe {x}: True')
    else:
        print(f'Missing values in dataframe {x}: False')
        
for x in dataset:
    sets = dataset[x]
    columns = list(sets.loc[:, sets.columns!= "id"].isna().sum()) #storing a variale to make the code easier to read
    print(f'Number of missing values in each column in {x}: {columns}')

for x in dataset:
    sets = dataset[x]
    sum_of_NaN = sum(list(sets.loc[:, sets.columns!= "id"].isna().sum()))
    print(f'total number of missing values in {x}: {sum_of_NaN}')

for x in dataset:
    sets = dataset[x]
    indexing = pd.isna(sets).any(1).to_numpy().nonzero()[0]
    objects = list(sets['id'][indexing])
    print(f'ID of rows with missing values in {x}: {objects}')

#Part 3
index_verdi = []
for x in dataset:
    sets = dataset[x]
    index_verdi.append(pd.isna(sets).any(1).to_numpy().nonzero()[0])
index_verdi = list(chain(*index_verdi))

print(index_verdi)

for y in dataset:
    dataset[y] = dataset[y].drop(index_verdi)
    
#part 4
sns.pairplot(dataset['A'])
rows_columns = dataset['A'].shape
print(f'amount of rows and columns in the dataset is: {rows_columns}')

dataset['B'].hist(figsize= (10,10))
rows_columns = dataset['B'].shape
print(f'Amount of rows and columns in the dataset is: {rows_columns}')

fig, ax = plt.subplots(figsize = (10,8))
sns.violinplot(data = dataset['C'], palette = "Set3", ax = ax, linewidth = 1)
rows_columns = dataset['C'].shape
print(f'Amount of rows and columns in the dataset is: {rows_columns}')

sns.heatmap(dataset['D'].corr(), linewidth = .1, vmax = 1, vmin = -1)
rows_columns = dataset['D'].shape
print(f'Amount of rows and columns in the dataset is: {rows_columns}')




