"""-*- coding: utf-8 -*-
 DateTime   : 2018/9/20 16:39
 Author  : Peter_Bonnie
 FileName    : data_show.py
 Software: PyCharm
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_cur_service_user_id(file,save_file):
    """
    获取最后的结果并提交
    :param file:
    :param save_file:
    :return:
    """
    data=pd.read_csv(file)
    train=pd.DataFrame()
    train['user_id']=data['user_id']
    train['current_service']=data['current_service']
    train.to_csv(save_file,index=False)

if __name__=="__main__":
    get_cur_service_user_id("result.csv",save_file="submit_94.csv")




# """预测后的值
# 90063345    116006
# 89950167     20679
# 89950166     17322
# 90109916      9899
# 99999828      8598
# 90155946      6913
# 99999826      5391
# 99999827      4307
# 99999830      3937
# 89950168      3757
# 99999825      3191
# """
#
#
# 99999825    71384
# 89950168    57302
# 99999828    41136
# 90155946    10324
# 90063345     5867
# 99999827     5566
# 99999826     4746
# 90109916     2279
# 89950166     1374
# 89950167       22
#
#

