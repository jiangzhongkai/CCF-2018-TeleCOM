import pandas as pd
import numpy as np

def process_null(df):
    df[['gender']] = pd.to_numeric(df.gender, errors='coerce')
    df[['age']] = pd.to_numeric(df.age, errors='coerce')
    df[['fee_1_month']] = pd.to_numeric(df.fee_1_month, errors='coerce')
    df[['fee_2_month']] = pd.to_numeric(df.fee_2_month, errors='coerce')
    df[['fee_3_month']] = pd.to_numeric(df.fee_3_month, errors='coerce')
    df[['fee_4_month']] = pd.to_numeric(df.fee_4_month, errors='coerce')
    df[['traffic_0_month']] = pd.to_numeric(df.traffic_0_month, errors='coerce')
    df[['traffic_1_month']] = pd.to_numeric(df.traffic_1_month, errors='coerce')
    df[['traffic_local_0_month']] = pd.to_numeric(df.traffic_local_0_month, errors='coerce')
    df[['contract_type']] = pd.to_numeric(df.contract_type, errors='coerce')
    df[['contract_time']] = pd.to_numeric(df.contract_time, errors='coerce')
    df[['call_local']] = pd.to_numeric(df.call_local, errors='coerce')
    df[['call_service_1_month']] = pd.to_numeric(df.call_service_1_month, errors='coerce')
    df[['call_service_2_month']] = pd.to_numeric(df.call_service_2_month, errors='coerce')
    df[['pay_num']] = pd.to_numeric(df.pay_num, errors='coerce')
    df[['pay_times']] = pd.to_numeric(df.pay_times, errors='coerce')
    df[['service_type']] = pd.to_numeric(df.service_type, errors='coerce')
    df[['is_mix_service']] = pd.to_numeric(df.is_mix_service, errors='coerce')
    df[['is_over_fee']] = pd.to_numeric(df.is_over_fee, errors='coerce')
    df[['complaint_level']] = pd.to_numeric(df.complaint_level, errors='coerce')
    df[['former_complaint_num']] = pd.to_numeric(df.former_complaint_num, errors='coerce')
    df[['former_complaint_fee']] = pd.to_numeric(df.former_complaint_fee, errors='coerce')
    df[['is_promise_low_consume']] = pd.to_numeric(df.is_promise_low_consume, errors='coerce')
    df[['net_service']] = pd.to_numeric(df.net_service, errors='coerce')
    df[['online_time']] = pd.to_numeric(df.online_time, errors='coerce')
    df[['mean_fee_month']]=pd.to_numeric(df.mean_fee_month,errors='coerce')
    df = df.replace(np.nan, -99999.99)
    return df


def save_data(df, file_name):
    df.to_csv(file_name, index=False)

def processing_data_na():
    """
    处理异常数据
    :return:
    """


if __name__ == '__main__':
    import time
    start_1=time.time()
    train = pd.read_csv("data/train.csv",low_memory=False)
    test = pd.read_csv("data/test.csv",low_memory=False)
    end_1=time.time()
    print("read data cost {} sec".format(str(round(end_1-start_1))))

    train.columns = ['service_type',
                     'is_mix_service',
                     'online_time',
                     'fee_1_month',
                     'fee_2_month',
                     'fee_3_month',
                     'fee_4_month',
                     'traffic_0_month',
                     'is_over_fee',
                     'contract_type',
                     'contract_time',
                     'is_promise_low_consume',
                     'net_service',
                     'pay_times',
                     'pay_num',
                     'traffic_1_month',
                     'traffic_local_0_month',
                     'call_local',
                     'call_service_1_month',
                     'call_service_2_month',
                     'gender',
                     'age',
                     'complaint_level',
                     'former_complaint_num',
                     'former_complaint_fee',
                     'current_service',
                     'user_id']
    test.columns = ['service_type',
                    'is_mix_service',
                    'online_time',
                    'fee_1_month',
                    'fee_2_month',
                    'fee_3_month',
                    'fee_4_month',
                    'traffic_0_month',
                    'is_over_fee',
                    'contract_type',
                    'contract_time',
                    'is_promise_low_consume',
                    'net_service',
                    'pay_times',
                    'pay_num',
                    'traffic_1_month',
                    'traffic_local_0_month',
                    'call_local',
                    'call_service_1_month',
                    'call_service_2_month',
                    'gender',
                    'age',
                    'complaint_level',
                    'former_complaint_num',
                    'former_complaint_fee',
                    'user_id']

    train = process_null(train)
    save_data(train, "data/train_pro_4.csv")
    test = process_null(test)
    save_data(test, "data/test_pro_4.csv")


