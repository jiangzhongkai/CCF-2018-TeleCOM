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
    # get_cur_service_user_id("result.csv",save_file="submit_94.csv")
    data=pd.read_csv("data/train.csv")
    print(data['current_service'].value_counts())
