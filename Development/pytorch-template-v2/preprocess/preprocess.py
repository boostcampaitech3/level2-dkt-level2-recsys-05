import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class SequencePreprocess:
    def __init__(self, bargs, data_dir):
        self.args = bargs
        self.data_dir = data_dir

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_split_data(self, oof = 0):
        val_user_list = self.oof_user_set[oof]
        train = []
        valid = []

        group_df = self.train_data.groupby('userID')

        for userID, df in group_df:
            if userID in val_user_list:
                trn_df = df.iloc[:-1, :]
                val_df = df.copy()
                train.append(trn_df)
                valid.append(val_df)
            else:
                train.append(df)

        train = pd.concat(train).reset_index(drop = True)
        valid = pd.concat(valid).reset_index(drop = True)
        
        return train, valid

    def split_user_set(self, all_df, oof = 5, seed = 22):
        user_list = all_df['userID'].unique().tolist()
        oof_user_set = {}
        kf = KFold(n_splits = oof, random_state = seed, shuffle = True)
        for idx, (train_user, valid_user) in enumerate(kf.split(user_list)):
            oof_user_set[idx] = valid_user.tolist()

        return oof_user_set

    def __preprocessing(self, df):

        # index 로 변환
        def get_val2idx(val_list : list) -> dict:
            val2idx = {}
            for idx, val in enumerate(val_list):
                val2idx[val] = idx
            
            return val2idx

        assessmentItemID2idx = get_val2idx(df['assessmentItemID'].unique().tolist())
        testId2idx = get_val2idx(df['testId'].unique().tolist())
        KnowledgeTag2idx = get_val2idx(df['KnowledgeTag'].unique().tolist())
        large_paper_number2idx = get_val2idx(df['large_paper_number'].unique().tolist())

        df['assessmentItemID2idx'] = df['assessmentItemID'].apply(lambda x : assessmentItemID2idx[x])
        df['testId2idx'] = df['testId'].apply(lambda x : testId2idx[x])
        df['KnowledgeTag2idx'] = df['KnowledgeTag'].apply(lambda x : KnowledgeTag2idx[x])
        df['large_paper_number2idx'] = df['large_paper_number'].apply(lambda x : large_paper_number2idx[x])

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.num_assessmentItemID = len(assessmentItemID2idx)
        self.args.num_testId = len(testId2idx)
        self.args.num_KnowledgeTag = len(KnowledgeTag2idx)
        self.args.num_large_paper_number = len(large_paper_number2idx)
        self.args.num_hour = 24
        self.args.num_dayofweek = 7
        self.args.num_week_number = 53

        return df

    def __feature_engineering(self, df):
        
        # 시험지 대분류
        def get_large_paper_number(x):
            return x[1:4]
        df['large_paper_number'] = df['assessmentItemID'].apply(lambda x : get_large_paper_number(x))

        # 문제 푸는데 걸린 시간
        def get_now_elapsed(df):
            
            diff = df.loc[:, ['userID','Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
            diff = diff.fillna(pd.Timedelta(seconds=0))
            diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
            df['now_elapsed'] = diff
            df['now_elapsed'] = df['now_elapsed'].apply(lambda x : x if x < 650 and x >=0 else 0)
            df['now_elapsed'] = df['now_elapsed']

            return df

        df = get_now_elapsed(df = df)

        # 문항별 정답률
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_mean_answerCode'] = df[df['answerCode'] != -1].groupby('assessmentItemID').mean()['answerCode']
        df = df.reset_index(drop = False)

        # 문항별 정답률 표준편차
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_std_answerCode'] = df[df['answerCode'] != -1].groupby('assessmentItemID').std()['answerCode']
        df = df.reset_index(drop = False)

        # 올바르게 푼 사람들의 문항별 풀이 시간 평균
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_mean_now_elapsed'] = df[df['answerCode'] == 1].groupby('assessmentItemID').mean()['now_elapsed']
        df = df.reset_index(drop = False)

        # 올바르게 푼 사람들의 문항별 풀이 시간 표준 편차
        df = df.set_index('assessmentItemID')
        df['assessmentItemID_std_now_elapsed'] = df[df['answerCode'] == 1].groupby('assessmentItemID').std()['now_elapsed']
        df = df.reset_index(drop = False)

        ## cat형
        # 문제 푼 시간
        df['hour'] = df['Timestamp'].dt.hour

        # 문제 푼 요일
        df['dayofweek'] = df['Timestamp'].dt.dayofweek

        # # 문제 푼 주
        # df['week_number'] = df['Timestamp'].apply(lambda x:x.isocalendar()[1]) - 1

        ## num 형
        # https://github.com/tidyverse/lubridate/issues/731
        # 문제 푼 시간
        sin_val = np.sin(2 * np.pi * np.array([i for i in range(1, 25)]) / 24)
        cos_val = np.cos(2 * np.pi * np.array([i for i in range(1, 25)]) / 24)

        df['num_hour'] = 2 * np.pi * (df['hour'] + 1) / 24
        df['sin_num_hour'] = np.sin(df['num_hour']) + 2 * abs(sin_val.min())
        df['cos_num_hour'] = np.cos(df['num_hour']) + 2 * abs(cos_val.min())

        # 문제 푼 요일
        sin_val = np.sin(2 * np.pi * np.array([i for i in range(1, 8)]) / 7)
        cos_val = np.cos(2 * np.pi * np.array([i for i in range(1, 8)]) / 7)

        df['num_dayofweek'] = 2 * np.pi * (df['dayofweek'] + 1) / 7
        df['sin_num_dayofweek'] = np.sin(df['num_dayofweek']) + 2 * abs(sin_val.min())
        df['cos_num_dayofweek'] = np.cos(df['num_dayofweek']) + 2 * abs(cos_val.min())

        # # 문제 푼 주
        # sin_val = np.sin(2 * np.pi * np.array([i for i in range(1, 54)]) / 53)
        # cos_val = np.cos(2 * np.pi * np.array([i for i in range(1, 54)]) / 53)
        # df['num_week_number'] = 2 * np.pi * (df['week_number'] + 1) / 53
        # df['sin_num_week_number'] = np.sin(df['num_week_number']) + 2 * abs(sin_val.min())
        # df['cos_num_week_number'] = np.cos(df['num_week_number']) + 2 * abs(cos_val.min())

        # 해당 대분류 시험지를 푼 기간 (주 단위)
        def get_now_week(df):
            userID2large_paper_number2week_number2now_week = {}
            group_df = df.groupby('userID')

            for userID, g_df in group_df:
                large_paper_number2week_number = {}
                gg_df = g_df.groupby('large_paper_number')
                for large_paper_number, ggg_df in gg_df:
                    week_number2now_week = {}
                    for idx, week_number in enumerate(sorted(ggg_df['week_number'].unique())):
                        week_number2now_week[week_number] = idx
                    
                    large_paper_number2week_number[large_paper_number] = week_number2now_week

                userID2large_paper_number2week_number2now_week[userID] = large_paper_number2week_number

            def get_now_week_val(x):
                return userID2large_paper_number2week_number2now_week[x['userID']][x['large_paper_number']][x['week_number']]

            df['now_week'] = df.apply(lambda x : get_now_week_val(x), axis = 1)

            return df

        # df = get_now_week(df = df)

        return df

    def load_data_from_file(self, data_dir, is_train=True):

        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
        }
        train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'), dtype=dtype, parse_dates=['Timestamp'])
        train_df = train_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'), dtype=dtype, parse_dates=['Timestamp'])
        test_df = test_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        test_user_set = test_df['userID'].unique().tolist()

        all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

        all_df = self.__feature_engineering(df = all_df)
        all_df = self.__preprocessing(df = all_df)
        self.oof_user_set = self.split_user_set(all_df)

        if is_train:
            df = all_df[all_df['answerCode'] != -1].reset_index(drop=True)
        else:
            df = all_df.set_index('userID').loc[test_user_set, :].reset_index(drop = False).sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        
        return df

    def load_train_data(self):
        self.train_data = self.load_data_from_file(self.data_dir)

    def load_test_data(self):
        self.test_data = self.load_data_from_file(self.data_dir, is_train=False)
