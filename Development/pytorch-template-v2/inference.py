import os
import argparse
from box import Box
from parse_config import ConfigParser

import torch
import numpy as np
import pandas as pd

import preprocess.preprocess as module_preprocess
import dataset.datasets as module_datasets
import data_loader.data_loaders as module_data_loaders
import model.model as module_arch


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.isdir(config["trainer"]['submission_dir']):
        os.mkdir(config["trainer"]['submission_dir'])
    
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
    preprocess_data.load_test_data()
    test_df = preprocess_data.get_test_data()
    print("preprocess complete!")

    test_dataset = config.init_obj("dataset", module_datasets, bargs=bridge_args, df = test_df)
    test_data_loader = config.init_obj("data_loader", module_data_loaders, dataset=test_dataset, is_train = False)

    model = config.init_obj("arch", module_arch, bargs=bridge_args)
    model = model.to(device)
    
    predict_list = []

    for oof in range(config['trainer']['oof']):
        model.load_state_dict(torch.load(os.path.join(config['trainer']['save_dir'], f'oof_{oof}_' + config['name'] + '.pt')))
        model.eval()
        predict = []
        with torch.no_grad():
            for data in test_data_loader:
                output = model(data)
                predict.extend(output[:, -1].cpu().numpy().tolist())

        predict_list.append(predict)
    
    predict_list = np.array(predict_list).mean(axis = 0)

    submission = pd.DataFrame(data = np.array(predict_list), columns = ['prediction'])
    submission['id'] = submission.index
    submission = submission[['id', 'prediction']]
    submission.to_csv(os.path.join(config["trainer"]['submission_dir'], 'OOF-Ensemble-' + config['name'] + '.csv'), index = False)
    print('Completion !')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
