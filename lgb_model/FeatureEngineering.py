

class FeatureEngineering:

    def __init__(self):
        self.df = None
        self.dft = None

    def feature_process(self, df, dft):
        self.df = FeatureEngineering.change_by_row(df)
        self.df = FeatureEngineering.remove_feature(self.df)
        self.dft = FeatureEngineering.change_by_row(dft)
        self.dft = FeatureEngineering.remove_feature(self.dft)

        return self.df, self.dft

    @staticmethod
    def change_by_row(df):
        fee_min = list()
        fee_max = list()
        fee_interval_min = list()
        fee_interval_max = list()
        traffic_max = list()
        traffic_min = list()
        traffic_sum = list()
        traffic_service = list()
        call_max = list()
        call_min = list()
        fee_mean_month=list()
        call_local_and_service = list()

        for row in range(df.shape[0]):

            #月度最低出账金额
            fee_min_item = min(df.at[row, 'fee_1_month'],
                               df.at[row, 'fee_2_month'],
                               df.at[row, 'fee_3_month'],
                               df.at[row, 'fee_4_month'])

            #月度最高出账金额
            fee_max_item = max(df.at[row, 'fee_1_month'],
                               df.at[row, 'fee_2_month'],
                               df.at[row, 'fee_3_month'],
                               df.at[row, 'fee_4_month'])
            #月度平均出账金额
            fee_mean_item=(df.at[row,'fee_1_month']+df.at[row,'fee_2_month']+df.at[row,'fee_3_month']+df.at[row,'fee_4_month'])/4.0

            fee_min.append(fee_min_item)
            fee_max.append(fee_max_item)
            fee_mean_month.append(fee_mean_item)

            #月度
            traffic_sum.append(df.at[row, 'traffic_0_month'] + df.at[row, 'traffic_1_month'])
            traffic_service.append(df.at[row, 'traffic_0_month'] - df.at[row, 'traffic_local_0_month'])

            if df.at[row, 'traffic_1_month'] > df.at[row, 'traffic_0_month']:
                traffic_max.append(df.at[row, 'traffic_1_month'])
                traffic_min.append(df.at[row, 'traffic_0_month'])
            else:
                traffic_max.append(df.at[row, 'traffic_0_month'])
                traffic_min.append(df.at[row, 'traffic_1_month'])

            if df.at[row, 'call_service_1_month'] > df.at[row, 'call_service_2_month']:
                call_max.append(df.at[row, 'call_service_1_month'])
                call_min.append(df.at[row, 'call_service_2_month'])
                call_local_and_service.append(df.at[row, 'call_service_1_month'] + df.at[row, 'call_local'])
            else:
                call_max.append(df.at[row, 'call_service_2_month'])
                call_min.append(df.at[row, 'call_service_1_month'])
                call_local_and_service.append(df.at[row, 'call_service_2_month'] + df.at[row, 'call_local'])

            if df.at[row, 'is_over_fee'] == 1:
                fee_interval_min.append(0.0)
                fee_interval_max.append(fee_min_item)
            else:
                fee_interval_min.append(fee_max_item)
                fee_interval_max.append(9999.0)

        df['fee_min'] = fee_min
        df['fee_max'] = fee_max
        df['fee_interval_min'] = fee_interval_min
        df['fee_interval_max'] = fee_interval_max
        df['traffic_max'] = traffic_max
        df['traffic_min'] = traffic_min
        df['traffic_sum'] = traffic_sum
        df['call_max'] = call_max
        df['call_min'] = call_min
        df['fee_mean_month']=fee_mean_month
        df['call_local_and_service'] = call_local_and_service

        return df

    @staticmethod
    def remove_feature(df):
        df = df.drop(['former_complaint_num',
                      # 'pay_times', #本次添加
                      'complaint_level',
                      'former_complaint_fee',
                      'is_over_fee',   #本次添加2018.10.6
                      # 'is_promise_low_consume',  #本次添加
                      'is_mix_service',
                      # "service_type",    #本次添加
                      'net_service'], axis=1)
        return df

