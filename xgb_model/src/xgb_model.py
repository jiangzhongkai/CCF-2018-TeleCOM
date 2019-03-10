import xgboost as xgb
# from Constant import Const
import operator
import pandas as pd
import time
from sklearn.metrics import f1_score
import numpy as np

#自定义评估函数
def f1_score(preds,valid):
    labels = valid.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')  # 改了下f1_score的计算方式
    return 'f1_score', score_vali, True


class xgb_model:

    def __init__(self, mode):
        self.params = {
                       'silent': 1,  #
                       'colsample_bytree': 0.8, #
                       'eval_metric': 'mlogloss',
                       'eta': 0.05,
                       'learning_rate': 0.1,
                       'njob': 8,
                       'min_child_weight': 1,
                       # 'subsample': 0.8,
                       'seed': 0,  #0
                       'objective': 'multi:softmax',
                       'max_depth': 6,#原来是6
                       'gamma': 0.0,
                       'booster': 'gbtree',
                       'num_class': 11
        }

        self.dtrain = None
        self.dvalid = None
        self.dtest = None
        self.model = None
        self.mode = mode

    def train_model(self, X_train, y_train, X_valid, y_valid, X_test, result=None):

        self.dtrain = xgb.DMatrix(X_train.drop(['user_id'], axis=1), y_train, missing=-99999.99)
        if self.mode:
            self.dvalid = xgb.DMatrix(X_valid.drop(['user_id'], axis=1), y_valid, missing=-99999.99)
            watchlist = [(self.dtrain, 'train'), (self.dvalid, 'valid')]
        else:
            self.dtest = xgb.DMatrix(X_test.drop(['user_id'], axis=1), missing=-99999.99)
            watchlist = [(self.dtrain, 'train')]

        self.model = xgb.train(self.params,
                               self.dtrain,
                               evals=watchlist,
                               num_boost_round=1000,#1000
                               # feval='f1_score',
                               early_stopping_rounds=100)
        #模型的保存

        if self.mode:
            self.valid_model(X_valid, y_valid)
        else:
            result = self.predict_model(X_test)

        importance = self.model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        pd.DataFrame(importance,columns=['feature','score']).to_csv('feature_importance.csv',index=False)
        print(pd.DataFrame(importance, columns=['feature', 'score']))
        return result

    def load_model(self):
        self.model = xgb.Booster({'nthread':8})
        self.model.load_model('../data/xgb_')
        self.valid_model()

    def valid_model(self, X_valid, y_valid):
        prediction = self.model.predict(self.dvalid)
        err = xgb_model.evaluation(prediction, X_valid, y_valid)
        print('the total f-score:', err)

    def predict_model(self, X_test):
        prediction = self.model.predict(self.dtest)
        result = xgb_model.test_result_merge(prediction, X_test)
        print('finished model training')
        xgb_model.save_data(result)
        return result

    @staticmethod
    def test_result_merge(predict_result, test):
        result = test
        result['current_service'] = predict_result
        result = result.reset_index(drop=True)
        result['current_service'] = result['current_service'].astype('float64')
        return result

    @staticmethod
    def valid_result_merge(predict_result, test, label):
        result = test
        result['current_service'] = predict_result
        result['label'] = label
        result = result.reset_index(drop=True)
        result['current_service'] = result['current_service'].astype('float64')
        return result

    @staticmethod
    def evaluation(predict_result, test, label):
        result = xgb_model.valid_result_merge(predict_result, test, label)
        score = 0.0
        service_list = result.groupby(['label']).count().index.values

        for service in service_list:
            tp = result[(result['label'] == result['current_service']) & (result['label'] == service)].shape[0]
            fp = result[(result['label'] != result['current_service']) & (result['current_service'] == service)].shape[0]
            fn = result[(result['label'] != result['current_service']) & (result['label'] == service)].shape[0]

            try:
                precision = float(tp)/(tp+fp)
                recall = float(tp)/(tp+fn)
                score += 2*precision*recall/(precision+recall)
            except ZeroDivisionError:
                print('zero error in service:', service)
                continue

        return (score/len(service_list))**2

    @staticmethod
    def save_data(result):
        submission = result[['user_id', 'current_service']]
        submission.columns = ['user_id', 'current_service']
        submission.to_csv("submit_2.csv", index=False)
        result.to_csv("result.csv",index=False)



