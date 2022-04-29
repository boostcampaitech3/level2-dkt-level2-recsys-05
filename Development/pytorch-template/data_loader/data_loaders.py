from torchvision import datasets as tvdatasets
from torchvision import transforms
import torch
from base import BaseDataLoader
import numpy as np
from preprocess import Preprocess

from torch.nn.utils.rnn import pad_sequence


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)
        # for el in cate_cols:
        #     print(el.shape)
        # print('-'*20)
        return cate_cols

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = tvdatasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class IscreamDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        bargs,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):

        # self.data_dir = "/opt/ml/input/data/"

        # TODO: preprocess 후, dataset생성 -> Dkt train.py 참조
        # args = Box(
        #     {
        #         "asset_dir" : "asset/",
        #         "data_dir" : "/opt/ml/input/data/",
        #         "file_name" : "train_data.csv",
        #         "max_seq_len" : 20
        #     }
        # )
        bargs.asset_dir = "asset/"
        bargs.data_dir = "/opt/ml/input/data/"
        bargs.file_name = "train_data.csv"
        bargs.max_seq_len = 20

        print("preprocessing iscream data...")
        preprocess = Preprocess(bargs)
        preprocess.load_train_data(bargs.file_name)
        train_data = preprocess.get_train_data()
        print("preprocess complete!")

        self.dataset = DKTDataset(train_data, bargs)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
