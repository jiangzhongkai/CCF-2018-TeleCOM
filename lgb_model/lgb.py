# 导入数据包
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import f1_score
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier,StackingClassifier
import matplotlib.pyplot as plt
from FeatureEngineering import *
from sklearn.externals import joblib

os.environ['CUDA_VISIBLE_DEVICES']='0'  #当没有相应的GPU设备时，会使用CPU来运行。
# 基础配置信息
path = 'data/'
n_splits = 8 #8
seed = 42

# lgb 参数
params={
    "learning_rate":0.2,
    "lambda_l1":0.1,
    "lambda_l2":0.2,
    "max_depth":6,  #6  本次修改了
    "objective":"multiclass",
    "num_class":11,
    "verbose":-1,
}
#
# 读取数据
train = pd.read_csv(path + 'train_pro_3.csv',low_memory=False)
train.drop_duplicates(inplace=True)
test = pd.read_csv(path + 'test_pro_3.csv',low_memory=False)
test.drop_duplicates(inplace=True)

#添加的部分================================================
feature=FeatureEngineering()
train,test=feature.feature_process(train,test)
print(train.columns)
#========================================================
#查看下年龄
print("age======")
print(train['age'].value_counts())
print(test['age'].value_counts())
#修改
# train.drop(train['age']=="-99999.99")
train['age']=train['age'].replace("\\N",0)
# train.drop(train[train['age']>100])
print("Afetr.....")
train['age']=train['age'].astype(np.int64)
test['age']=test['age'].astype(np.int64)
print(train['age'].value_counts())
#查看性别
print("gender=====")
print(train['gender'].value_counts())
print(test['gender'].value_counts())

#性别这里还没有处理好0-表示男性   1-表示女性
train['gender']=train['gender'].replace("1",1).astype(int)
train['gender']=train['gender'].replace("0",0).astype(int)
train['gender']=train['gender'].replace("00",0).astype(int)
train['gender']=train['gender'].replace("01",1).astype(int)
train['gender']=train['gender'].replace("02",2).astype(int)
train['gender']=train['gender'].replace("2",2).astype(int)
train['gender']=train['gender'].replace("\\N",0).astype(int)
print("After....")
print(train['gender'].value_counts())
print(test['gender'].value_counts())

# 对标签编码 映射关系.
label2current_service = dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))

print(label2current_service)
print(current_service2label)
# 原始数据的标签映射,
train['current_service'] = train['current_service'].map(current_service2label)

# 构造原始数据,作为目标预测
y = train.pop('current_service')
train_id = train.pop('user_id')
# 这个字段有点问题
X = train
train_col = train.columns


test_id = test['user_id']
test.pop('user_id')
X_test = test[test.columns]
print("X_test.shape:",X_test.shape)
# 数据有问题数据
for i in train_col:
    X[i] = X[i].replace("\\N",-1)
for i in test.columns:
    X_test[i] = X_test[i].replace("\\N",-1)

X,y,X_test = X.values,y,X_test.values

#
print("X",X)
print("X_test",X_test)

print("X的形状是:",X.shape)
count_1=0
count_2=0
for i in range(X.shape[0]):
    if isinstance(X[i,21],str):
        count_1+=1
        X[i,21]=int(X[i,21])
    else:
        count_2+=1
print("count_1:",count_1)
print("count_2:",count_2)

count_3=0
count_4=0
for i in range(X_test.shape[0]):
    if isinstance(X_test[i,21],str):
        count_3+=1
        X_test[i,21]=int(X_test[i,21])
    else:
        count_4+=1

print("count_3:",count_3)
print("count_4:",count_4)

# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')   #改了下f1_score的计算方式
    return 'f1_score', score_vali, True

xx_score = []
cv_pred = []

#先对模型进行调参
#lgb模型,k折交叉验证，分类问题使用分层抽样
skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
import time
now=time.time()
for index,(train_index,test_index) in enumerate(skf.split(X,y)):
    X_train,X_valid,y_train,y_valid = X[train_index],X[test_index],y[train_index],y[test_index]
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)
    clf=lgb.train(params,train_data,num_boost_round=10000,valid_sets=[validation_data],early_stopping_rounds=200,feval=f1_score_vali,verbose_eval=1)
    plt.figure(figsize=(12,6))
    lgb.plot_importance(clf, max_num_features=40)
    plt.title("Featurertances")
    plt.show()
    feature_importance=pd.DataFrame({
         'column': train_col,
         'importance': clf.feature_importance(),
     }).to_csv('feature_importance_leaves57.csv',index=False)
    xx_pred = clf.predict(X_valid,num_iteration=clf.best_iteration)
    xx_pred = [np.argmax(x) for x in xx_pred]
    xx_score.append(f1_score(y_valid,xx_pred,average='macro'))
    y_test = clf.predict(X_test,num_iteration=clf.best_iteration)
    y_test = [np.argmax(x) for x in y_test]  #输出概率最大的那个
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))


# 其实这里已经对8折的数据做了一次投票，最后输出投票后的结果
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

# 保存结果
df_test = pd.DataFrame()
df_test['user_id'] = list(test_id.unique())
df_test['current_service'] = submit
df_test['current_service'] = df_test['current_service'].map(label2current_service)
df_test.to_csv("baseline_25.csv",index=False)

print(xx_score,np.mean(xx_score)**2)
end=time.time()
print("total cost {} second ".format(str(round(end-now,2))))  #总的花费了多少时间


