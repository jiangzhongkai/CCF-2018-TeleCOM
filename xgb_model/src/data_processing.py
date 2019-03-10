import pandas as pd
from sklearn.model_selection import GroupKFold,StratifiedKFold

class data_processing:

    def __init__(self, mode):
        self.df = None
        self.dft = None
        self.services = list()
        self.mode = mode
        self.n_splits=8
        self.seed=42

    def data_input(self, train_file_name, test_file_name):
        self.df = pd.read_csv(train_file_name,low_memory=False)
        self.df = self.label_process(self.df)
        print('the shape of train data:', self.df.shape)

        if not self.mode:
            self.dft = pd.read_csv(test_file_name,low_memory=False)
            print('the shape of test data:', self.dft.shape)
        else:
            self.dft = None

        return self.df, self.dft

    def label_process(self, data_frame):
        self.services = data_frame.groupby(['current_service']).count().index.values
        for label_id in range(11):
            service = self.services[label_id]
            data_frame['current_service'] = data_frame['current_service'].replace(service, label_id)

        return data_frame


    def get_split_data(self, df_bin, dft_bin):
        user_id = [user[0] for user in df_bin[['user_id']].values]
        # group = GroupKFold(n_splits=10)
        skr=StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

        for train_idx, valid_idx in skr.split(df_bin.drop(['current_service'], axis=1),
                                                    df_bin[['current_service']],
                                                    groups=user_id):
            train = df_bin.iloc[train_idx]
            valid = df_bin.iloc[valid_idx]
            X_train = train.drop(['current_service'], axis=1)
            y_train = train[['current_service']]
            X_valid = valid.drop(['current_service'], axis=1)
            y_valid = valid[['current_service']]
            break

        if not self.mode:
            X_train = df_bin.drop(['current_service'], axis=1)
            y_train = df_bin[['current_service']]

        X_test = dft_bin

        return X_train, y_train, X_valid, y_valid, X_test

    def transform_index(self, result):
        for label_id in range(11):
            service = self.services[label_id]
            result['current_service'] = result['current_service'].replace(label_id, int(service))
        result = result[['user_id', 'current_service']]
        result.to_csv('../data/submission_18.csv', index=False)
