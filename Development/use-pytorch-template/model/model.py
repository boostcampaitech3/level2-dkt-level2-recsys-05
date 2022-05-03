import torch
import torch.nn as nn

from .layer import SASRecBlock

class LSTMATTN(nn.Module):
    def __init__(self, bargs, hidden_dim, embedding_size, num_heads, num_layers, dropout_rate):
        super(LSTMATTN, self).__init__()
        self.args = bargs
        
        self.num_assessmentItemID = self.args.num_assessmentItemID
        self.num_testId = self.args.num_testId
        self.num_KnowledgeTag = self.args.num_KnowledgeTag
        self.num_large_paper_number = self.args.num_large_paper_number
        self.num_hour = self.args.num_hour
        self.num_dayofweek = self.args.num_dayofweek
        self.num_week_number = self.args.num_week_number
        self.cat_cols = self.args.cat_cols
        self.num_cols = self.args.num_cols

        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Embedding
        emb = {}
        for cat_col in self.cat_cols:
            if cat_col == 'assessmentItemID2idx':
                emb[cat_col] = nn.Embedding(self.num_assessmentItemID + 1, self.embedding_size, padding_idx = 0) # 문항에 대한 정보
            elif cat_col == 'testId2idx':
                emb[cat_col] = nn.Embedding(self.num_testId + 1, self.embedding_size, padding_idx = 0) # 시험지에 대한 정보
            elif cat_col == 'KnowledgeTag2idx':
                emb[cat_col] = nn.Embedding(self.num_KnowledgeTag + 1, self.embedding_size, padding_idx = 0) # 지식 태그에 대한 정보
            elif cat_col == 'large_paper_number2idx':
                emb[cat_col] = nn.Embedding(self.num_large_paper_number + 1, self.embedding_size, padding_idx = 0) # 학년에 대한 정보
            elif cat_col == 'hour':
                emb[cat_col] = nn.Embedding(self.num_hour + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 시간에 대한 정보
            elif cat_col == 'dayofweek':
                emb[cat_col] = nn.Embedding(self.num_dayofweek + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 요일에 대항 정보
            elif cat_col == 'week_number':
                emb[cat_col] = nn.Embedding(self.num_week_number + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 주에 대항 정보

        self.emb_dict = nn.ModuleDict(emb)

        self.answerCode_emb = nn.Embedding(3, self.embedding_size, padding_idx = 0) # 문제 정답 여부에 대한 정보

        # embedding combination projection
        self.cat_emb = nn.Sequential(
            nn.Linear((len(self.cat_cols) + 1) * self.embedding_size, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.num_emb = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = self.dropout_rate,
            )

        # Attention
        self.blocks = nn.ModuleList([SASRecBlock(self.num_heads, self.hidden_dim, self.dropout_rate) for _ in range(self.num_layers)])

        # predict
        self.dropout = nn.Dropout(self.dropout_rate)

        # Fully connected layer
        self.predict_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        past_cat_feature : (batch_size, max_len, cat_cols)
        past_num_feature : (batch_size, max_len, num_cols)
        past_answerCode : (batch_size, max_len)

        now_cat_feature : (batch_size, max_len, cat_cols)
        now_num_feature : (batch_size, max_len, num_cols)
        
        """

        past_answerCode = input['past_answerCode']
        now_cat_feature = input['now_cat_feature'].to(self.device)
        now_num_feature = input['now_num_feature'].to(self.device)

        # masking 
        mask_pad = torch.BoolTensor(past_answerCode > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, past_answerCode.size(1), past_answerCode.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to(self.device) # (batch_size, 1, max_len, max_len)

        # embedding
        cat_emb_list = [self.answerCode_emb(past_answerCode.to(self.device))]
        for idx, cat_col in enumerate(self.cat_cols):
            cat_emb_list.append(self.emb_dict[cat_col](now_cat_feature[:, :, idx]))

        cat_emb = torch.concat(cat_emb_list, dim = -1)
        cat_emb = self.cat_emb(cat_emb)
        
        num_emb = self.num_emb(now_num_feature)

        emb = torch.concat([cat_emb, num_emb], dim = -1)

        # LSTM
        emb, _ = self.lstm(emb)
        
        # Attention
        for block in self.blocks:
            emb, attn_dist = block(emb, mask)

        # predict
        output = self.predict_layer(self.dropout(emb))

        return output.squeeze(2)


class LSTM(nn.Module):
    def __init__(self, bargs, hidden_dim, embedding_size, num_layers, dropout_rate):
        super(LSTM, self).__init__()
        self.args = bargs
        self.num_assessmentItemID = self.args.num_assessmentItemID
        self.num_testId = self.args.num_testId
        self.num_KnowledgeTag = self.args.num_KnowledgeTag
        self.num_large_paper_number = self.args.num_large_paper_number
        self.num_hour = self.args.num_hour
        self.num_dayofweek = self.args.num_dayofweek
        self.num_week_number = self.args.num_week_number
        self.cat_cols = self.args.cat_cols
        self.num_cols = self.args.num_cols

        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Embedding
        emb = {}
        for cat_col in self.cat_cols:
            if cat_col == 'assessmentItemID2idx':
                emb[cat_col] = nn.Embedding(self.num_assessmentItemID + 1, self.embedding_size, padding_idx = 0) # 문항에 대한 정보
            elif cat_col == 'testId2idx':
                emb[cat_col] = nn.Embedding(self.num_testId + 1, self.embedding_size, padding_idx = 0) # 시험지에 대한 정보
            elif cat_col == 'KnowledgeTag2idx':
                emb[cat_col] = nn.Embedding(self.num_KnowledgeTag + 1, self.embedding_size, padding_idx = 0) # 지식 태그에 대한 정보
            elif cat_col == 'large_paper_number2idx':
                emb[cat_col] = nn.Embedding(self.num_large_paper_number + 1, self.embedding_size, padding_idx = 0) # 학년에 대한 정보
            elif cat_col == 'hour':
                emb[cat_col] = nn.Embedding(self.num_hour + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 시간에 대한 정보
            elif cat_col == 'dayofweek':
                emb[cat_col] = nn.Embedding(self.num_dayofweek + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 요일에 대항 정보
            elif cat_col == 'week_number':
                emb[cat_col] = nn.Embedding(self.num_week_number + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 주에 대항 정보

        self.emb_dict = nn.ModuleDict(emb)

        self.answerCode_emb = nn.Embedding(3, self.embedding_size, padding_idx = 0) # 문제 정답 여부에 대한 정보

        # embedding combination projection
        self.cat_emb = nn.Sequential(
            nn.Linear((len(self.cat_cols) + 1) * self.embedding_size, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.num_emb = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.lstm = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = self.dropout_rate,
            )

        # predict
        self.dropout = nn.Dropout(self.dropout_rate)

        # Fully connected layer
        self.predict_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        past_cat_feature : (batch_size, max_len, cat_cols)
        past_num_feature : (batch_size, max_len, num_cols)
        past_answerCode : (batch_size, max_len)

        now_cat_feature : (batch_size, max_len, cat_cols)
        now_num_feature : (batch_size, max_len, num_cols)
        
        """

        past_answerCode = input['past_answerCode'].to(self.device)
        now_cat_feature = input['now_cat_feature'].to(self.device)
        now_num_feature = input['now_num_feature'].to(self.device)

        # past
        cat_emb_list = [self.answerCode_emb(past_answerCode)]
        for idx, cat_col in enumerate(self.cat_cols):
            cat_emb_list.append(self.emb_dict[cat_col](now_cat_feature[:, :, idx]))

        cat_emb = torch.concat(cat_emb_list, dim = -1)
        cat_emb = self.cat_emb(cat_emb)
        
        num_emb = self.num_emb(now_num_feature)

        emb = torch.concat([cat_emb, num_emb], dim = -1)

        emb, _ = self.lstm(emb)

        output = self.predict_layer(self.dropout(emb))

        return output.squeeze(2)


class TransformerAndLSTMEncoderDeocoderEachEmbedding(nn.Module):
    def __init__(self, bargs, hidden_dim, embedding_size, num_heads, num_layers, dropout_rate):
        super(TransformerAndLSTMEncoderDeocoderEachEmbedding, self).__init__()

        self.args = bargs
        
        self.num_assessmentItemID = self.args.num_assessmentItemID
        self.num_testId = self.args.num_testId
        self.num_KnowledgeTag = self.args.num_KnowledgeTag
        self.num_large_paper_number = self.args.num_large_paper_number
        self.num_hour = self.args.num_hour
        self.num_dayofweek = self.args.num_dayofweek
        self.num_week_number = self.args.num_week_number
        self.cat_cols = self.args.cat_cols
        self.num_cols = self.args.num_cols

        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # past
        past_emb = {}
        for cat_col in self.cat_cols:
            if cat_col == 'assessmentItemID2idx':
                past_emb[cat_col] = nn.Embedding(self.num_assessmentItemID + 1, self.embedding_size, padding_idx = 0) # 문항에 대한 정보
            elif cat_col == 'testId2idx':
                past_emb[cat_col] = nn.Embedding(self.num_testId + 1, self.embedding_size, padding_idx = 0) # 시험지에 대한 정보
            elif cat_col == 'KnowledgeTag2idx':
                past_emb[cat_col] = nn.Embedding(self.num_KnowledgeTag + 1, self.embedding_size, padding_idx = 0) # 지식 태그에 대한 정보
            elif cat_col == 'large_paper_number2idx':
                past_emb[cat_col] = nn.Embedding(self.num_large_paper_number + 1, self.embedding_size, padding_idx = 0) # 학년에 대한 정보
            elif cat_col == 'hour':
                past_emb[cat_col] = nn.Embedding(self.num_hour + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 시간에 대한 정보
            elif cat_col == 'dayofweek':
                past_emb[cat_col] = nn.Embedding(self.num_dayofweek + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 요일에 대항 정보
            elif cat_col == 'week_number':
                past_emb[cat_col] = nn.Embedding(self.num_week_number + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 주에 대항 정보

        self.past_emb_dict = nn.ModuleDict(past_emb)

        self.past_answerCode_emb = nn.Embedding(3, self.hidden_dim, padding_idx = 0) # 문제 정답 여부에 대한 정보

        self.past_cat_emb = nn.Sequential(
            nn.Linear(len(self.cat_cols) * self.embedding_size, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.past_num_emb = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.emb_layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        self.past_lstm = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = self.dropout_rate,
            )

        self.past_blocks = nn.ModuleList([SASRecBlock(self.num_heads, self.hidden_dim, self.dropout_rate) for _ in range(self.num_layers)])

        # now

        now_emb = {}
        for cat_col in self.cat_cols:
            if cat_col == 'assessmentItemID2idx':
                now_emb[cat_col] = nn.Embedding(self.num_assessmentItemID + 1, self.embedding_size, padding_idx = 0) # 문항에 대한 정보
            elif cat_col == 'testId2idx':
                now_emb[cat_col] = nn.Embedding(self.num_testId + 1, self.embedding_size, padding_idx = 0) # 시험지에 대한 정보
            elif cat_col == 'KnowledgeTag2idx':
                now_emb[cat_col] = nn.Embedding(self.num_KnowledgeTag + 1, self.embedding_size, padding_idx = 0) # 지식 태그에 대한 정보
            elif cat_col == 'large_paper_number2idx':
                now_emb[cat_col] = nn.Embedding(self.num_large_paper_number + 1, self.embedding_size, padding_idx = 0) # 학년에 대한 정보
            elif cat_col == 'hour':
                now_emb[cat_col] = nn.Embedding(self.num_hour + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 시간에 대한 정보
            elif cat_col == 'dayofweek':
                now_emb[cat_col] = nn.Embedding(self.num_dayofweek + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 요일에 대항 정보
            elif cat_col == 'week_number':
                now_emb[cat_col] = nn.Embedding(self.num_week_number + 1, self.embedding_size, padding_idx = 0) # 문제 풀이 주에 대항 정보

        self.now_emb_dict = nn.ModuleDict(now_emb)

        self.now_cat_emb = nn.Sequential(
            nn.Linear(len(self.cat_cols) * self.embedding_size, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.now_num_emb = nn.Sequential(
            nn.Linear(len(self.num_cols), self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2, eps=1e-6)
        )

        self.now_lstm = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = False,
            dropout = self.dropout_rate,
            )

        self.now_blocks = nn.ModuleList([SASRecBlock(self.num_heads, self.hidden_dim, self.dropout_rate) for _ in range(self.num_layers)])

        # predict

        self.dropout = nn.Dropout(self.dropout_rate)

        self.predict_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    
    def forward(self, input):
        """
        past_cat_feature : (batch_size, max_len, cat_cols)
        past_num_feature : (batch_size, max_len, num_cols)
        past_answerCode : (batch_size, max_len)

        now_cat_feature : (batch_size, max_len, cat_cols)
        now_num_feature : (batch_size, max_len, num_cols)
        
        """

        past_cat_feature = input['past_cat_feature'].to(self.device)
        past_num_feature = input['past_num_feature'].to(self.device) 
        past_answerCode = input['past_answerCode']
        now_cat_feature = input['now_cat_feature'].to(self.device)
        now_num_feature = input['now_num_feature'].to(self.device)

        # masking 
        mask_pad = torch.BoolTensor(past_answerCode > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        mask_time = (1 - torch.triu(torch.ones((1, 1, past_answerCode.size(1), past_answerCode.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        mask = (mask_pad & mask_time).to(self.device) # (batch_size, 1, max_len, max_len)

        # past
        past_cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            past_cat_emb_list.append(self.past_emb_dict[cat_col](past_cat_feature[:, :, idx]))

        past_cat_emb = torch.concat(past_cat_emb_list, dim = -1)
        past_cat_emb = self.past_cat_emb(past_cat_emb)
        past_num_emb = self.past_num_emb(past_num_feature)

        past_emb = torch.concat([past_cat_emb, past_num_emb], dim = -1)
        past_emb += self.past_answerCode_emb(past_answerCode.to(self.device))
        past_emb = self.emb_layernorm(past_emb) # LayerNorm

        for block in self.past_blocks:
            past_emb, attn_dist = block(past_emb, mask)

        past_emb, _ = self.past_lstm(past_emb)

        # now
        now_cat_emb_list = []
        for idx, cat_col in enumerate(self.cat_cols):
            now_cat_emb_list.append(self.now_emb_dict[cat_col](now_cat_feature[:, :, idx]))

        now_cat_emb = torch.concat(now_cat_emb_list, dim = -1)
        now_cat_emb = self.now_cat_emb(now_cat_emb)
        now_num_emb = self.now_num_emb(now_num_feature)

        now_emb = torch.concat([now_cat_emb, now_num_emb], dim = -1)

        for block in self.now_blocks:
            now_emb, attn_dist = block(now_emb, mask)

        now_emb, _ = self.now_lstm(now_emb)
        
        # predict
        emb = torch.concat([past_emb, now_emb], dim = -1)
        output = self.predict_layer(self.dropout(emb))

        return output.squeeze(2)