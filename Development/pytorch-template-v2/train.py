import os
import argparse
import importlib
import collections
from box import Box
from parse_config import ConfigParser

import torch
import random
import numpy as np

import preprocess.preprocess as module_preprocess
import dataset.datasets as module_datasets
import data_loader.data_loaders as module_data_loaders
import model.loss as module_loss
import model.model as module_arch

# fix random seeds for reproducibility
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.isdir(config["trainer"]['save_dir']):
        os.mkdir(config["trainer"]['save_dir'])

    # 데이터와 모델 연결되는 args
    bridge_args = Box({
        'cat_cols' : ['assessmentItemID2idx', 'testId2idx', 'KnowledgeTag2idx', 'large_paper_number2idx', 'hour', 'dayofweek'], 
        'num_cols' : ['now_elapsed', 'assessmentItemID_mean_now_elapsed', 'assessmentItemID_std_now_elapsed', 'assessmentItemID_mean_answerCode', 'assessmentItemID_std_answerCode']
        })
    
    # bridge_args = Box({
    #     'cat_cols' : config['cat_cols'], 
    #     'num_cols' : config['num_cols'],
    #     })


    print("preprocessing data...")
    preprocess_data = config.init_obj("preprocess", module_preprocess, bargs=bridge_args)
    preprocess_data.load_train_data()
    print("preprocess complete!")

    for oof in range(config['trainer']['oof']):
        seed_everything(config['trainer']['seed'])
        train_df, valid_df = preprocess_data.get_split_data(oof)

        train_dataset = config.init_obj("dataset", module_datasets, bargs=bridge_args, df = train_df)
        valid_dataset = config.init_obj("dataset", module_datasets, bargs=bridge_args, df = valid_df)

        train_data_loader = config.init_obj("data_loader", module_data_loaders, dataset=train_dataset)
        valid_data_loader = config.init_obj("data_loader", module_data_loaders, dataset=valid_dataset, is_train = False)

        model = config.init_obj("arch", module_arch, bargs=bridge_args)
        model = model.to(device)

        criterion = getattr(module_loss, config["loss"])
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj("optimizer", torch.optim, trainable_params)

        trainer_module = getattr(
            importlib.import_module(f'trainer.{config["trainer"]["type"]}'),
            config["trainer"]["type"],
        )

        trainer = trainer_module(
            model,
            criterion,
            optimizer,
            config=config,
            device=device,
            data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
        )

        trainer.train(oof)

        print(f'BEST OOF-{oof}| Epoch: {trainer.best_epoch:3d}| loss: {trainer.best_loss:.5f}| acc: {trainer.best_acc:.5f}| roc_auc: {trainer.best_roc_auc:.5f}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    main(config)
