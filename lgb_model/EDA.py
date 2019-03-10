# EDA BASE

import pandas as pd
import numpy as np
from pandas import DataFrame as DF
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate missing values by columns
def missing_values_table(data,topK=20):
    mis_val = data.isnull().sum()
    mis_val_percent = 100 * data.isnull().sum() / len(data)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1]!=0].sort_values(
    '% of Total Values', ascending=False).round(2)
    
    print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    
    print(mis_val_table_ren_columns.head(topK))

# Function to print columns types
def column_types_table(data):
	print('Number of each type of columns:')
	count_dtype = DF(data.dtypes.value_counts()).reset_index()
	count_dtype.columns = ['name','total']
	print(count_dtype)

	print('\nNumber of unique classes in each columns:')
	for i in count_dtype['name'].values:
		print('Type: ',i)
		print(DF(data.select_dtypes(i).apply(pd.Series.nunique, axis=0)).sort_values(by=[0],ascending=False).rename(columns={0:'NUNIQUE'}))

# Function to screen big diffrence feature between Train and Test
def train_test_distribution_difference(train_data,test_data,ignored_col=[],min_threshold=0.8,max_threshold=1.25):
	print('Screening big difference between train and test: ')
	print('Min Threshold: ',min_threshold,' \nMax Threshold: ',max_threshold)
	object_col = train_data.select_dtypes('object').columns
	numerical_col = list(set(train_data.columns)-set(object_col)-set(ignored_col))
	print('Numerical Length:' ,len(numerical_col))
	train_des = train_data[numerical_col].describe().T
	test_des = test_data[numerical_col].describe().T
	calc_diff = train_des/test_des
	std_calc = calc_diff[(calc_diff['std']>=min_threshold) & (calc_diff['std']<=max_threshold)]
	print('Std feature length: ',std_calc.shape[0])
	print('Std cover: \n',std_calc.index.values)
	mean_calc = calc_diff[(calc_diff['mean']>=min_threshold) & (calc_diff['mean']<=max_threshold)]
	print('Mean feature length: ',mean_calc.shape[0])
	print('Mean cover: \n',mean_calc.index.values)
	both = list(set(std_calc.index.values)&set(mean_calc.index.values))
	print('Both mean std: ',len(both))
	print('Both cover: \n',both)
	union = list(set(std_calc.index.values)|set(mean_calc.index.values))
	print('Union mean std: ',len(union))
	print('Union cover: \n',union)
	# Return 4 Seq
	return std_calc.index.values,mean_calc.index.values,both,union

# Function to analysis label describe
def label_analysis(data,label_name=None,feature_name=[]):
	print('LABEL CATEGORY Analysis')
	count_label = DF(data[label_name].value_counts()).reset_index()
	count_label.columns = ['cate','total']
	print(count_label)
	try:
		data[label_name].astype(int).plot.hist()
		plt.show()
	except:
		data[label_name].fillna(-1).astype(int).plot.hist()
		plt.show()
    # Describe 01
	if len(feature_name)==0:
		feature_name = [i for i in data.columns if i not in [label_name,]]
	print('Want To Watch: ',len(feature_name))
	print(feature_name)
	print('Describe in each columns: ')
	for i in count_label['cate'].values:
		print('Cate: ',i)
		print(data[data[label_name].astype(int)==i][feature_name].describe())

	print('CALC CORR')
	correlations = data.corr()[label_name].sort_values()
	print('Most Positive Correlations:\n', correlations.tail(15))
	print('\nMost Negative Correlations:\n', correlations.head(15))

if __name__=="__main__":
    data=pd.read_csv("data/train.csv")
    label_analysis(data)






