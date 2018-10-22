import pandas as pd
import numpy as np
import time
import math

"""
这个文件主要是处理特征以及特征工程的选取
"""
start=time.time()
print("Loading data,please wait a monent.")
train=pd.read_csv("../data/train.csv",low_memory=False)
train.drop_duplicates(inplace=True)
test=pd.read_csv("../data/test.csv",low_memory=False)
temp_train=pd.read_csv("../data/train.csv",low_memory=False)
temp_test=pd.read_csv("../data/test.csv",low_memory=False)
###

end=time.time()
print("loading data has been fininshed,and its costs {} sec!!!".format(str(round(end-start,2))))

#出账金额权重
temp_train['2_total_fee']=temp_train['2_total_fee'].replace("\\N",-1)
temp_train['3_total_fee']=temp_train['3_total_fee'].replace("\\N",-1)
temp_train['2_total_fee']=temp_train['2_total_fee'].astype(np.float64)
temp_train['3_total_fee']=temp_train['3_total_fee'].astype(np.float64)

temp_test['2_total_fee']=temp_test['2_total_fee'].replace("\\N",-1)
temp_test['3_total_fee']=temp_test['3_total_fee'].replace("\\N",-1)
temp_test['2_total_fee']=temp_test['2_total_fee'].astype(np.float64)
temp_test['3_total_fee']=temp_test['3_total_fee'].astype(np.float64)

total_fee_weight_train=0.8*temp_train['1_total_fee']+0.5*temp_train['2_total_fee']\
                                  +0.2*temp_train['3_total_fee']+temp_train['4_total_fee']
print(total_fee_weight_train)
total_fee_weight_test=0.8*temp_test['1_total_fee']+0.5*temp_test['2_total_fee']\
                                  +0.2*temp_test['3_total_fee']+temp_test['4_total_fee']
print(total_fee_weight_test)

train.insert(0,column='fee_total_weight',value=total_fee_weight_train)
test.insert(0,column='fee_total_weight',value=total_fee_weight_test)

#缴费金额与缴费次数
temp_train['pay_num']=temp_train['pay_num'].replace("\\N",0)
temp_test['pay_num']=temp_test['pay_num'].replace("\\N",0)
temp_train['pay_times']=temp_train['pay_times'].replace("\\N",1)
temp_test['pay_times']=temp_test['pay_times'].replace("\\N",1)

temp_train['pay_num']=temp_train['pay_num'].astype(np.float64)
temp_test['pay_num']=temp_test['pay_num'].astype(np.float64)

pay_num_times_train=temp_train['pay_num']/temp_train['pay_times']
pay_num_times_test=temp_test['pay_num']/temp_test['pay_times']
#
train.insert(1,column='pay_num_times',value=pay_num_times_train)
test.insert(1,column='pay_num_times',value=pay_num_times_test)

train = train.replace(np.nan, -99999.99)
test=test.replace(np.nan,-99999.99)


#月度最低出账金额
temp_min_fee_month_train=[]
temp_min_fee_month_test=[]

fee_inter_max_train=[]
fee_inter_min_train=[]
fee_inter_max_test=[]
fee_inter_min_test=[]
for i in range(temp_train.shape[0]):
    # print(temp_train['3_total_fee'][i])
    temp_min_fee_month_train.append(min(temp_train['1_total_fee'][i],temp_train['2_total_fee'][i],
                                        temp_train['3_total_fee'][i],temp_train['4_total_fee'][i]))

    if temp_train['many_over_bill'][i]==1:
        fee_inter_max_train.append(min(temp_train['1_total_fee'][i],temp_train['2_total_fee'][i],
                                        temp_train['3_total_fee'][i],temp_train['4_total_fee'][i]))
        fee_inter_min_train.append(0.0)
    else:
        fee_inter_min_train.append(max(temp_train['1_total_fee'][i], temp_train['2_total_fee'][i],
                                    temp_train['3_total_fee'][i], temp_train['4_total_fee'][i]))
        fee_inter_max_train.append(9999.0)


print(temp_test.shape[0])
for j in range(temp_test.shape[0]):
    # print(temp_test['2_total_fee'][j])
    temp_min_fee_month_test.append(min(temp_test['1_total_fee'][j],temp_test['2_total_fee'][j],\
                                       temp_test['3_total_fee'][j],temp_test['4_total_fee'][j]))
    if temp_test['many_over_bill'][j]==1:
        fee_inter_max_test.append(min(temp_test['1_total_fee'][j],temp_test['2_total_fee'][j],\
                                        temp_test['3_total_fee'][j],temp_test['4_total_fee'][j]))
        fee_inter_min_test.append(0.0)
    else:
        fee_inter_min_test.append(max(temp_test['1_total_fee'][j], temp_test['2_total_fee'][j],\
                                    temp_test['3_total_fee'][j], temp_test['4_total_fee'][j]))
        fee_inter_max_test.append(9999.0)

train.insert(2,column='min_fee_month',value=temp_min_fee_month_train)
test.insert(2,column='min_fee_month',value=temp_min_fee_month_test)

train.insert(10,column='fee_inter_max',value=fee_inter_max_train)
train.insert(11,column='fee_inter_min',value=fee_inter_min_train)

test.insert(10,column='fee_inter_max',value=fee_inter_max_test)
test.insert(11,column='fee_inter_min',value=fee_inter_min_test)

#月度最高出帐金额
temp_max_fee_month_train=[]
temp_max_fee_month_test=[]

#月度平均出账金额
temp_mean_fee_month_train=[]
temp_mean_fee_month_test=[]

for i in range(temp_train.shape[0]):
    temp_max_fee_month_train.append(max(temp_train['1_total_fee'][i],temp_train['2_total_fee'][i],
                                        temp_train['3_total_fee'][i],temp_train['4_total_fee'][i]))

    temp_mean_fee_month_train.append((temp_train['1_total_fee'][i]+temp_train['2_total_fee'][i]+
                                        temp_train['3_total_fee'][i]+temp_train['4_total_fee'][i])/4.0)

for j in range(temp_test.shape[0]):
    temp_max_fee_month_test.append(max(temp_test['1_total_fee'][j],temp_test['2_total_fee'][j],
                                       temp_test['3_total_fee'][j],temp_test['4_total_fee'][j]))
    temp_mean_fee_month_test.append((temp_test['1_total_fee'][j]+temp_test['2_total_fee'][j]+
                                       temp_test['3_total_fee'][j]+temp_test['4_total_fee'][j])/4.0)

#插入到csv文件中
train.insert(3,column='max_fee_month',value=temp_max_fee_month_train)
test.insert(3,column='max_fee_month',value=temp_max_fee_month_test)

train.insert(12,column='mean_fee_month',value=temp_mean_fee_month_train)
test.insert(12,column='mean_fee_month',value=temp_mean_fee_month_test)

#主要是计算当前与上月的流量之和
traffic_sum_train=[]
traffic_sum_test=[]
for i in range(temp_train.shape[0]):
    traffic_sum_train.append(temp_train['month_traffic'][i]+temp_train['last_month_traffic'][i])
for j in range(temp_test.shape[0]):
    traffic_sum_test.append(temp_test['month_traffic'][j]+temp_test['last_month_traffic'][j])

train.insert(4,column='traffic_month_sum',value=traffic_sum_train)
test.insert(4,column='traffic_month_sum',value=traffic_sum_test)

#月流量最大值和月流量最小值
traffic_max_month_train=[]
traffic_max_month_test=[]

traffic_min_month_train=[]
traffic_min_month_test=[]

for i in range(temp_train.shape[0]):
    if temp_train['month_traffic'][i]>temp_train['last_month_traffic'][i]:
        traffic_max_month_train.append(temp_train['month_traffic'][i])
        traffic_min_month_train.append(temp_train['last_month_traffic'][i])
    else:
        traffic_max_month_train.append(temp_train['last_month_traffic'][i])
        traffic_min_month_train.append(temp_train['month_traffic'][i])

for j in range(temp_test.shape[0]):
    if temp_test['month_traffic'][j]>temp_test['last_month_traffic'][j]:
        traffic_max_month_test.append(temp_test['month_traffic'][j])
        traffic_min_month_test.append(temp_test['last_month_traffic'][j])
    else:
        traffic_max_month_test.append(temp_test['last_month_traffic'][j])
        traffic_min_month_test.append(temp_test['month_traffic'][j])



train.insert(5,column='traffic_month_max',value=traffic_max_month_train)
train.insert(6,column='traffic_month_min',value=traffic_min_month_train)

test.insert(5,column='traffic_month_max',value=traffic_max_month_test)
test.insert(6,column='traffic_month_min',value=traffic_min_month_test)

#计算通话时长的最大值和最小值以及本地通话时长
call_min_train=[]
call_min_test=[]
call_max_train=[]
call_max_test=[]
call_local_service_train=[]
call_local_service_test=[]
for i in range(temp_train.shape[0]):
    if temp_train['service1_caller_time'][i]>temp_train['service2_caller_time'][i]:
        call_max_train.append(temp_train['service1_caller_time'][i])
        call_min_train.append(temp_train['service2_caller_time'][i])
        call_local_service_train.append(temp_train['service1_caller_time'][i]+temp_train['local_caller_time'][i])
    else:
        call_max_train.append(temp_train['service2_caller_time'][i])
        call_min_train.append(temp_train['service1_caller_time'][i])
        call_local_service_train.append(temp_train['service2_caller_time'][i] + temp_train['local_caller_time'][i])
for i in range(temp_test.shape[0]):
    if temp_test['service1_caller_time'][i] > temp_test['service2_caller_time'][i]:
        call_max_test.append(temp_test['service1_caller_time'][i])
        call_min_test.append(temp_test['service2_caller_time'][i])
        call_local_service_test.append(temp_test['service1_caller_time'][i] + temp_test['local_caller_time'][i])
    else:
        call_max_test.append(temp_test['service2_caller_time'][i])
        call_min_test.append(temp_test['service1_caller_time'][i])
        call_local_service_test.append(temp_test['service2_caller_time'][i] + temp_test['local_caller_time'][i])

train.insert(7,column='call_max',value=call_max_train)
train.insert(8,column='call_min',value=call_min_train)
train.insert(9,column='call_local_service',value=call_local_service_train)

test.insert(7,column='call_max',value=call_max_test)
test.insert(8,column='call_min',value=call_min_test)
test.insert(9,column='call_local_service',value=call_local_service_test)
#继续做特征地处

#保存到csv文件中
train.to_csv("../data/train_pro_4.csv",index=False)
test.to_csv("../data/test_pro_4.csv",index=False)

print("Processing fininshe and its cost:{} sec".format(str(round(end-start,2))))
print(train.info())
print(test.info())

