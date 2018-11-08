from data_processing import *
from xgb_model import *
from TimeCost import *
from sklearn.model_selection import StratifiedKFold
from RemoveFeatures import *
from sklearn.preprocessing  import MinMaxScaler
import pandas as pd

if __name__ == '__main__':


    mode = True
    mode = False
    tc = TimeCost()
    dp = data_processing(mode)
    df, dft = dp.data_input('../data/train_pro_4.csv', '../data/test_pro_4.csv')

    tc.print_event()

    df=df.replace("\\N",-1)
    dft=dft.replace("\\N",-1)
    #对性别和年龄进行处理
    if 'age' in df.columns:
        #做相应的处理
        df['age']=df['age'].replace("\\N",-1)
        df['age']=df['age'].astype(np.int)
        dft['age']=dft['age'].astype(np.int)

    if  '2_total_fee' in df.columns and '3_total_fee' in df.columns:
        df['2_total_fee']=df['2_total_fee'].astype(np.float64)
        df['3_total_fee']=df['3_total_fee'].astype(np.float64)

        dft['2_total_fee'] = dft['2_total_fee'].astype(np.float64)
        dft['3_total_fee'] = dft['3_total_fee'].astype(np.float64)

    if 'gender' in df.columns:
        #做相应的处理
        df['gender'] = df['gender'].replace("1", 1).astype(int)
        df['gender'] = df['gender'].replace("0", 0).astype(int)
        df['gender'] = df['gender'].replace("00", 0).astype(int)
        df['gender'] = df['gender'].replace("01", 1).astype(int)
        df['gender'] = df['gender'].replace("02", 2).astype(int)
        df['gender'] = df['gender'].replace("2", 2).astype(int)
        df['gender'] = df['gender'].replace("\\N", 0).astype(int)

    print("df.columns:",df.columns)
    print("dft.columns:",dft.columns)

    rm_feats = RemoveFeat()

    #对离散数据进行one_hot编码
    cate_features=['is_promise_low_consume','contract_time','contract_type','many_over_bill','online_time','is_mix_service','service_type','net_service','complaint_level','gender','current_service','user_id']

    const_features = ['1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic','pay_times',\
                      'pay_num','last_month_traffic','local_trafffic_month','local_caller_time','service1_caller_time',\
                      'service2_caller_time','age','former_complaint_num','former_complaint_fee']

    #在做归一化和编码得时候，可以把训练集和测试集拼接在一起做，最后再把分割开.....
    #本次去掉相关的特征2018.10.14号
    # df=rm_feats.remove_feats(df,['former_complaint_num',
    #                   # 'pay_times',   #本次添加
    #                   'complaint_level',
    #                   'former_complaint_fee',
    #                   'many_over_bill',
    #                   # 'is_promise_low_consume',  #本次添加
    #                   #'service_type',  # 添加了
    #                   'is_mix_service',
    #                   #"gender",    #去掉gender试试
    #                   'net_service'])
    # dft=rm_feats.remove_feats(dft,['former_complaint_num',
    #                   # 'pay_times',   #本次添加
    #                   'complaint_level',
    #                   'former_complaint_fee',
    #                    'many_over_bill',
    #                   # 'is_promise_low_consume',  #本次添加
    #                   #'service_type',  # 添加了
    #                   'is_mix_service',
    #                   #"gender",    #去掉gender试试
    #                   'net_service'])
    df = rm_feats.remove_feats(df, ['former_complaint_num',
                                    'pay_times',   #本次添加
                                    # 'complaint_level',
                                    'former_complaint_fee',
                                    'many_over_bill',
                                    # 'is_promise_low_consume',  # 本次添加
                                    'service_type',  # 添加了
                                    # 'is_mix_service',
                                    # "gender",    #去掉gender试试
                                    'net_service',
                                    'service1_caller_time'])
    dft = rm_feats.remove_feats(dft,['former_complaint_num',
                                    'pay_times',   #本次添加
                                    # 'complaint_level',
                                    'former_complaint_fee',
                                    'many_over_bill',
                                    # 'is_promise_low_consume',  # 本次添加
                                    'service_type',  # 添加了
                                    # 'is_mix_service',
                                    # "gender",    #去掉gender试试
                                    'net_service',
                                    'service1_caller_time'])


    #给类别变量添加标签，转化为连续的值
    print("dft:",dft.columns)
    print("df:",df.columns)
    tc.print_event()

    #这里用了两种方法进行交叉验证，使用分层抽样，则flag=True,否则组验证，则flag=False
    flag=False
    if flag:
        #进行参数配置
        n_splits=5
        random_state=42
        shuffle=True
        skr=StratifiedKFold(n_splits=n_splits,random_state=random_state,shuffle=shuffle)
        y=df[['current_service']]
        X=df.drop(['current_service'],axis=1)
        # X_train,y_train,X_valid,y_valid,X_test=dp.get_split_data_1(df,dft,skr)  #获得交叉验证后的数据
        for index,(train_index,valid_index) in enumerate(skr.split(X,y)):
            x_train = X.iloc[train_index]
            x_valid = X.iloc[valid_index]
            y_train=y.iloc[train_index]
            y_valid=y.iloc[valid_index]
            x_test=dft
            xgb=xgb_model(mode)
            result=xgb.train_model(x_train,y_train,x_valid,y_valid,x_test)
            if index == 0:
                cv_pred = np.array(result).reshape(-1, 1)
            else:
                cv_pred = np.hstack((cv_pred, np.array(result).reshape(-1, 1)))
    else:
        X_train, y_train, X_valid, y_valid, X_test = dp.get_split_data(df, dft)
        tc.print_event()

        xgb = xgb_model(mode)
        result = xgb.train_model(X_train, y_train, X_valid, y_valid, X_test)
        tc.print_event()

        if not mode:
            dp.transform_index(result)
    # xgb.load_model()
    #投票
    # if len(cv_pred)!=0:
    #     submit = []
    #     for line in cv_pred:
    #         submit.append(np.argmax(np.bincount(line)))  # 统计出现的次数

        #提交结果
        # dp.transform_index(submit)

    #最后的结束时间
    # tc.print_event()


