"""-*- coding: utf-8 -*-
 DateTime   : 2018/10/2 23:38
 Author  : Peter_Bonnie
 FileName    : get_result.py
 Software: PyCharm
"""
import pandas as pd
import numpy as np

"""
获取最终的结果
"""
def get_final_result(data,to_file):
    result = pd.read_csv(data, low_memory=False)
    final_result = pd.DataFrame()
    final_result['user_id'] = result['user_id']
    final_result['current_service'] = result['current_service'].astype(np.int64)
    final_result.to_csv(to_file, index=False)


if __name__=="__main__":
    get_final_result("../data/submission_18.csv","baseline_128.csv")







