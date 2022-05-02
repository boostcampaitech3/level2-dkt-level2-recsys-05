from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from base import collate_fn as collate_fn_set

default_collate = default_collate

class TransformerDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        num_workers,
        collate_fn="transformer_collate",
        is_train = True,
    ):
        self.shuffle = shuffle

        self.collate_fn = collate_fn
        self.collate_fn = getattr(collate_fn_set, collate_fn)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
        }

        if is_train:
            self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn,
            "num_workers": num_workers,
        }
        
        super().__init__(**self.init_kwargs)
