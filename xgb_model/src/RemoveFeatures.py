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
from sklearn.preprocessing import MinMaxScaler



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

    @staticmethod
    def one_hot(df,cate_feats=[]):
        """
        对离散特征数据进行one-hot编码
        :param df:
        :param cate_feats:
        :return:
        """
        for col in cate_feats:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)
        df.drop(columns=cate_feats, inplace=True)

        return df

    def Max_Min(self,df,const_feats=[]):
        """
        对连续特征数据进行归一化操作
        :param df:
        :param const_feats:
        :return:
        """
        for col in const_feats:
            scaler=MinMaxScaler()
            df[col]=scaler.fit_transform(np.array(df[col].values.tolist()).reshape(-1,1))  #都是这样写的
        return df

if __name__=="__main__":
    data=pd.read_csv("../data/train_pro_3.csv")
    print(data['gender'].value_counts())
    print(data['gender'].dtypes)

