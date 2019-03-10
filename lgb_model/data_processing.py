import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import time
from sklearn.model_selection import  train_test_split
from FeatureEngineering import *
#数据的导入

start=time.time()
print("Loading Data....")
train=pd.read_csv("data/train_pro_2.csv",low_memory=False)
train.drop_duplicates(inplace=True)
test=pd.read_csv("data/test_pro_2.csv",low_memory=False)
temp_train=pd.read_csv("data/train.csv",low_memory=False)
temp_test=pd.read_csv("data/test.csv",low_memory=False)

end=time.time()
temp_train['2_total_fee']=temp_train['2_total_fee'].replace("\\N",-1)
temp_train['3_total_fee']=temp_train['3_total_fee'].replace("\\N",-1)
temp_train['2_total_fee']=temp_train['2_total_fee'].astype(np.float64)
temp_train['3_total_fee']=temp_train['3_total_fee'].astype(np.float64)

temp_test['2_total_fee']=temp_test['2_total_fee'].replace("\\N",-1)
temp_test['3_total_fee']=temp_test['3_total_fee'].replace("\\N",-1)
temp_test['2_total_fee']=temp_test['2_total_fee'].astype(np.float64)
temp_test['3_total_fee']=temp_test['3_total_fee'].astype(np.float64)

print(temp_train.dtypes)

#缴费金额与缴费次数
temp_train['pay_num']=temp_train['pay_num'].replace("\\N",0)
temp_test['pay_num']=temp_test['pay_num'].replace("\\N",0)
temp_train['pay_times']=temp_train['pay_times'].replace("\\N",1)
temp_test['pay_times']=temp_test['pay_times'].replace("\\N",1)

temp_train['pay_num']=temp_train['pay_num'].astype(np.float64)
temp_test['pay_num']=temp_test['pay_num'].astype(np.float64)

pay_num_times_train=temp_train['pay_num']/temp_train['pay_times']
pay_num_times_test=temp_test['pay_num']/temp_test['pay_times']
print(pay_num_times_train)
print(pay_num_times_test)
train.insert(1,column='pay_num_times',value=pay_num_times_train)
test.insert(1,column='pay_num_times',value=pay_num_times_test)

total_fee_weight_train=0.8*temp_train['1_total_fee']+0.5*temp_train['2_total_fee']\
                                  +0.2*temp_train['3_total_fee']+temp_train['4_total_fee']
print(total_fee_weight_train)
total_fee_weight_test=0.8*temp_test['1_total_fee']+0.5*temp_test['2_total_fee']\
                                  +0.2*temp_test['3_total_fee']+temp_test['4_total_fee']

train.insert(0,column='fee_total_weight',value=total_fee_weight_train)
test.insert(0,column='fee_total_weight',value=total_fee_weight_test)

#月度平均出账金额
temp_mean_fee_month_train=[]
temp_mean_fee_month_test=[]
for i in range(temp_train.shape[0]):
    temp_mean_fee_month_train.append((temp_train['1_total_fee'][i] + temp_train['2_total_fee'][i] +
                                      temp_train['3_total_fee'][i] + temp_train['4_total_fee'][i]) / 4.0)

for j in range(temp_test.shape[0]):
    temp_mean_fee_month_test.append((temp_test['1_total_fee'][j] + temp_test['2_total_fee'][j] +
                                         temp_test['3_total_fee'][j] + temp_test['4_total_fee'][j]) / 4.0)


train.insert(1,column='mean_fee_month',value=temp_mean_fee_month_train)
test.insert(1,column='mean_fee_month',value=temp_mean_fee_month_test)

train.to_csv("data/train_pro_3.csv",index=False)
test.to_csv("data/test_pro_3.csv",index=False)

print("Processing fininshe and its cost:{} sec".format(str(round(end-start,2))))
print(train.info())
print(test.info())


