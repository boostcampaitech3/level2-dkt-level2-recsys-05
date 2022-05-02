from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    def __init__(
        self,
        bargs,
        df,
        max_len = None,
        window = None,
        data_augmentation = False,
        ):

        self.args = bargs
        self.cat_cols = self.args.cat_cols
        self.num_cols = self.args.num_cols
        self.get_df = df.groupby('userID')
        self.user_list = df['userID'].unique().tolist()
        self.max_len = max_len
        self.window = window
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.cat_feature_list, self.num_feature_list, self.answerCode_list = self._data_augmentation()


    def __len__(self):
        if self.data_augmentation:
            return len(self.cat_feature_list)
        return len(self.user_list)

    def __getitem__(self, idx):
        if self.data_augmentation:
            cat_feature = self.cat_feature_list[idx]
            num_feature = self.num_feature_list[idx]
            answerCode = self.answerCode_list[idx]

            now_cat_feature = cat_feature[1:, :]
            now_num_feature = num_feature[1:, :]
            now_answerCode = answerCode[1:]
            
            past_cat_feature = cat_feature[:-1, :]
            past_num_feature = num_feature[:-1, :]
            past_answerCode = answerCode[:-1]
            
        else:
            user = self.user_list[idx]
            if self.max_len:
                get_df = self.get_df.get_group(user).iloc[-self.max_len:, :]
            else:
                get_df = self.get_df.get_group(user)

            now_df = get_df.iloc[1:, :]
            now_cat_feature = now_df[self.cat_cols].values
            now_num_feature = now_df[self.num_cols].values
            now_answerCode = now_df['answerCode'].values

            past_df = get_df.iloc[:-1, :]
            past_cat_feature = past_df[self.cat_cols].values
            past_num_feature = past_df[self.num_cols].values
            past_answerCode = past_df['answerCode'].values

        return {
            'past_cat_feature' : past_cat_feature, 
            'past_num_feature' : past_num_feature, 
            'past_answerCode' : past_answerCode, 
            'now_cat_feature' : now_cat_feature, 
            'now_num_feature' : now_num_feature, 
            'now_answerCode' : now_answerCode
            }
    

    def _data_augmentation(self):
        cat_feature_list = []
        num_feature_list = []
        answerCode_list = []
        for userID, get_df in self.get_df:
            cat_feature = get_df[self.cat_cols].values[::-1]
            num_feature = get_df[self.num_cols].values[::-1]
            answerCode = get_df['answerCode'].values[::-1]

            start_idx = 0

            if len(get_df) <= self.max_len:
                cat_feature_list.append(cat_feature[::-1])
                num_feature_list.append(num_feature[::-1])
                answerCode_list.append(answerCode[::-1])
            else:
                while True:
                    if len(cat_feature[start_idx: start_idx + self.max_len, :]) < self.max_len:
                        cat_feature_list.append(cat_feature[start_idx: start_idx + self.max_len, :][::-1])
                        num_feature_list.append(num_feature[start_idx: start_idx + self.max_len, :][::-1])
                        answerCode_list.append(answerCode[start_idx: start_idx + self.max_len][::-1])
                        break
                    cat_feature_list.append(cat_feature[start_idx: start_idx + self.max_len, :][::-1])
                    num_feature_list.append(num_feature[start_idx: start_idx + self.max_len, :][::-1])
                    answerCode_list.append(answerCode[start_idx: start_idx + self.max_len][::-1])
                    start_idx += self.window
            
        return cat_feature_list, num_feature_list, answerCode_list
