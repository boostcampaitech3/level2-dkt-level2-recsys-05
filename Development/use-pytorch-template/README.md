# PyTorch Template Project

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── inference.py - inference of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │
  ├── data_loader/ - anything about data loading goes here
  │
  ├── model/ - models, losses, and metrics
  │
  ├── trainer/ - trainers
  │
  ├── logger/ - module for tensorboard visualization and logging
  │  
  └── utils/ - small utility functions
  ```

## Usage
train
`python -W ignore train.py -c {모델 config 파일 명}`

inference
`python -W ignore inference.py -c {모델 config 파일 명}`

### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "모델이름",
    "entity" : "wandb project 만든 사람 계정명",
    
    "cat_cols" : ["사용할 cat cols"], 
    "num_cols" : ["사용할 num cols"],


    "preprocess": {
        "type": "사용할 Preprocess",
        "args": {
            "data_dir": "/opt/ml/input/data"
        }
    },

    "dataset" : {
        "type": "사용할 Dataset",
        "args": {
            "data_augmentation" : false
        }
    },

    "data_loader" : {
        "type": "사용할 DataLoader",
        "args": {
            "batch_size" : 32,
            "shuffle" : true,
            "num_workers" : 8,
            "collate_fn" : "시용할 collate"
        }
    },

    "arch" : {
        "type": "사용할 모델",
        "args": {
            "hidden_dim" : 128,
            "embedding_size" : 64,
            "num_heads" : 2,
            "num_layers" : 1,
            "dropout_rate" : 0.5
        }
    },

    "loss" : "bce_loss",

    "optimizer" : {
        "type": "Adam",
        "args":{
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": false
        }
    },

    "trainer" : {
        "type" : "사용할 Trainer",
        "epochs": 1,
        "oof" : 5,
        "seed" : 22,
        "save_dir": "/opt/ml/model",
        "submission_dir": "/opt/ml/submission" 
    }
}
```

Add addional configurations if you need.

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
