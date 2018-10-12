"""-*- coding: utf-8 -*-
 DateTime   : 2018/10/7 16:01
 Author  : Peter_Bonnie
 FileName    : remove_feat.py
 Software: PyCharm
"""
"""
删除不用的特征的类
"""
import numpy as np
import pandas as pd


class RemoveFeat(object):
    """
    delete some features
    """
    def __init__(self):
        """
        简单的初始化函数
        """

    def remove_feats(self,df,rm_list=list()):
        """
        :return:
        """
        self.df=df.drop(rm_list,axis=1)
        return self.df


if __name__=="__main__":
    data=pd.read_csv("../data/train_pro_3.csv")
    print(data['gender'].value_counts())
    print(data['gender'].dtypes)

