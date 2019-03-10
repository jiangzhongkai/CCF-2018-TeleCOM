# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#简单的投票融合
def vote(sublist=[]):
    result = pd.read_csv(sublist[0]).sort_values('user_id')
    print(result.columns)
    # print(result['current_service'])
    label2current_service = dict(
        zip(range(0, len(set(result['current_service']))), sorted(list(set(result['current_service'])))))
    current_service2label = dict(
        zip(sorted(list(set(result['current_service']))), range(0, len(set(result['current_service'])))))

    for i in sublist:
        temp = pd.read_csv(i).sort_values('user_id')
        result[i] = temp.current_service.map(current_service2label)
    temp_df = result[sublist]
    # 投票
    submit = []
    for line in temp_df.values:
        submit.append(np.argmax(np.bincount(line)))

    result['current_service'] = submit
    result['current_service'] = result['current_service'].map(label2current_service)
    result.to_csv('result.csv',index=False)
    return result[['user_id', 'current_service']]


data = vote(sublist=["submit_75.csv","submit_92.csv","submit_88.csv"])



