from torchvision import datasets as tvdatasets
from torchvision import transforms
import torch
from base import BaseDataLoader
import numpy as np
from preprocess import LSTMPreprocess


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, bargs):
        self.data = data
        self.args = bargs

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
        return cate_cols

    def __len__(self):
        return len(self.data)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        collate_fn="default_collate",
        bargs=None,
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
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )


class IscreamDataLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size,
        bargs,
        asset_dir,
        data_dir,
        file_name,
        collate_fn="sequence_collate",
        max_seq_len=20,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):

        bargs.asset_dir = asset_dir
        bargs.data_dir = data_dir
        bargs.file_name = file_name
        bargs.max_seq_len = max_seq_len

        print("preprocessing iscream data...")
        preprocess = LSTMPreprocess(bargs)
        preprocess.load_train_data(bargs.file_name)
        train_data = preprocess.get_train_data()
        print("preprocess complete!")

        self.dataset = DKTDataset(train_data, bargs)
        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )
